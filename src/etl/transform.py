import warnings
import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
log = get_logger(__name__)




def _clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, deduplicate, engineer delivery and time features."""
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    before = len(df)
    df = df.drop_duplicates("order_id").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        log.warning(f"  orders: removed {removed} duplicate rows")

    # Delivery metrics
    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days
    df["fulfillment_days"] = (
        df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["is_late_delivery"] = (df["delivery_delay_days"] > 0).astype("Int8")

    # Time features 
    ts = df["order_purchase_timestamp"]
    df["order_year"]  = ts.dt.year
    df["order_month"] = ts.dt.month
    df["order_week"]  = ts.dt.isocalendar().week.astype("Int32")
    df["order_dow"]   = ts.dt.day_name()
    df["order_hour"]  = ts.dt.hour
    df["order_date"]  = ts.dt.date.astype(str)
    df["year_week"]   = ts.dt.strftime("%Y-W%U")

    log.info(f"  orders cleaned  → {len(df):,} rows | "
             f"late deliveries: {df['is_late_delivery'].sum():,}")
    return df


def _clean_order_items(df: pd.DataFrame, min_price: float) -> pd.DataFrame:
    """Parse types, compute derived price columns, remove invalid rows."""
    df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")
    df["price"]         = pd.to_numeric(df["price"],         errors="coerce").fillna(0.0)
    df["freight_value"] = pd.to_numeric(df["freight_value"], errors="coerce").fillna(0.0)

    before = len(df)
    df = df[df["price"] >= min_price].reset_index(drop=True)
    if len(df) < before:
        log.warning(f"  order_items: removed {before - len(df)} rows with price < {min_price}")

    df["total_value"]   = df["price"] + df["freight_value"]
    df["freight_ratio"] = (df["freight_value"] / df["total_value"].replace(0, np.nan)).round(4)

    log.info(f"  order_items     → {len(df):,} rows | "
             f"GMV: R${df['price'].sum():,.0f}")
    return df


def _clean_payments(df: pd.DataFrame) -> pd.DataFrame:
    df["payment_value"] = pd.to_numeric(df["payment_value"], errors="coerce").fillna(0.0)
    df = df[df["payment_value"] > 0].reset_index(drop=True)
    log.info(f"  payments        → {len(df):,} rows")
    return df


def _clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df["review_creation_date"]    = pd.to_datetime(df["review_creation_date"],    errors="coerce")
    df["review_answer_timestamp"] = pd.to_datetime(df["review_answer_timestamp"], errors="coerce")
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
    df = df.dropna(subset=["review_score"]).reset_index(drop=True)
    df["is_negative"] = (df["review_score"] <= 2).astype("Int8")
    df["is_positive"] = (df["review_score"] >= 4).astype("Int8")
    log.info(f"  reviews         → {len(df):,} rows | "
             f"avg score: {df['review_score'].mean():.2f}")
    return df


def _clean_products(df: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        categories[["product_category_name", "product_category_name_english"]],
        on="product_category_name",
        how="left",
    )
    df["product_category_name_english"] = df["product_category_name_english"].fillna("other")
    log.info(f"  products        → {len(df):,} rows")
    return df



def _build_fact_orders(
    orders:   pd.DataFrame,
    customers:pd.DataFrame,
    items:    pd.DataFrame,
    payments: pd.DataFrame,
    reviews:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Master denormalised table joining all order-level data.
    One row per order.
    """
    item_agg = (
        items.groupby("order_id")
        .agg(
            item_count    =("order_item_id", "count"),
            order_revenue =("price",         "sum"),
            order_freight =("freight_value", "sum"),
            order_gmv     =("total_value",   "sum"),
        )
        .reset_index()
    )

    pay_agg = (
        payments.groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_type =("payment_type",  "first"),
            installments =("payment_installments", "max"),
        )
        .reset_index()
    )

    rev_agg = (
        reviews.groupby("order_id")
        .agg(
            review_score=("review_score", "mean"),
            is_negative =("is_negative",  "max"),
        )
        .reset_index()
    )

    fact = (
        orders
        .merge(customers[["customer_id", "customer_state", "customer_city"]],
               on="customer_id", how="left")
        .merge(item_agg,  on="order_id", how="left")
        .merge(pay_agg,   on="order_id", how="left")
        .merge(rev_agg,   on="order_id", how="left")
    )

    fact["order_revenue"]  = fact["order_revenue"].fillna(0.0)
    fact["order_freight"]  = fact["order_freight"].fillna(0.0)
    fact["freight_pct"]    = (
        fact["order_freight"] / fact["order_gmv"].replace(0, np.nan) * 100
    ).round(2)

    log.info(f"  fact_orders     → {len(fact):,} rows | "
             f"total GMV: R${fact['order_gmv'].sum():,.0f}")
    return fact


def _build_weekly_agg(fact: pd.DataFrame) -> pd.DataFrame:
    """Weekly time-series aggregation used by anomaly detection."""
    delivered = fact[fact["order_status"] == "delivered"].copy()
    delivered["week_start"] = (
        delivered["order_purchase_timestamp"]
        .dt.to_period("W")
        .apply(lambda p: p.start_time)
    )
    weekly = (
        delivered.groupby("week_start")
        .agg(
            order_count  =("order_id",        "count"),
            total_revenue=("order_revenue",    "sum"),
            avg_ticket   =("order_revenue",    "mean"),
            avg_review   =("review_score",     "mean"),
            late_count   =("is_late_delivery", "sum"),
            avg_freight  =("order_freight",    "mean"),
        )
        .reset_index()
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    log.info(f"  weekly_agg      → {len(weekly):>4} weeks")
    return weekly


def _build_seller_scorecard(
    items:   pd.DataFrame,
    orders:  pd.DataFrame,
    reviews: pd.DataFrame,
    sellers: pd.DataFrame,
    weights: dict,
    min_orders: int,
) -> pd.DataFrame:
    """
    Composite seller health scorecard.
    score = review_weight*(avg_review/5) + ontime_weight*(1-late_rate)
            + volume_weight*min(orders/50, 1)
    All weights from config.
    """
    merged = (
        items
        .merge(orders[["order_id","order_status","is_late_delivery"]], on="order_id", how="left")
        .merge(reviews[["order_id","review_score","is_negative"]],     on="order_id", how="left")
    )

    sc = (
        merged.groupby("seller_id")
        .agg(
            total_orders      =("order_id",        "nunique"),
            total_revenue     =("price",            "sum"),
            avg_price         =("price",            "mean"),
            avg_review_score  =("review_score",     "mean"),
            negative_review_pct=("is_negative",     "mean"),
            late_delivery_pct =("is_late_delivery", "mean"),
        )
        .reset_index()
    )
    sc = sc[sc["total_orders"] >= min_orders].copy()

    rw = weights["review"] / 100
    ow = weights["ontime"] / 100
    vw = weights["volume"] / 100

    sc["seller_health_score"] = (
        sc["avg_review_score"].fillna(3) / 5.0 * rw * 100
        + (1 - sc["late_delivery_pct"].fillna(0)) * ow * 100
        + sc["total_orders"].clip(upper=50) / 50 * vw * 100
    ).round(1)

    sc["total_revenue"]        = sc["total_revenue"].round(2)
    sc["avg_price"]            = sc["avg_price"].round(2)
    sc["avg_review_score"]     = sc["avg_review_score"].round(3)
    sc["negative_review_pct"]  = (sc["negative_review_pct"] * 100).round(2)
    sc["late_delivery_pct"]    = (sc["late_delivery_pct"]   * 100).round(2)

    sc = sc.merge(sellers[["seller_id","seller_state"]], on="seller_id", how="left")
    log.info(f"  seller_scorecard→ {len(sc):>4} sellers (min {min_orders} orders)")
    return sc



def transform(raw: dict[str, pd.DataFrame], cfg: dict = None) -> dict[str, pd.DataFrame]:
    """
    Run all transformations on the raw extracted DataFrames.

    Parameters
    ----------
    raw : dict  Output from extract()
    cfg : dict  Project config

    Returns
    -------
    dict of cleaned and derived DataFrames
    """
    cfg  = cfg or load_config()
    etl  = cfg["etl"]
    min_price = etl["min_price"]

    log.info("─" * 50)
    log.info("STAGE 2 — TRANSFORM")
    log.info("─" * 50)

    clean: dict[str, pd.DataFrame] = {}

    clean["orders"]      = _clean_orders(raw["orders"].copy())
    clean["order_items"] = _clean_order_items(raw["order_items"].copy(), min_price)
    clean["payments"]    = _clean_payments(raw["payments"].copy())
    clean["reviews"]     = _clean_reviews(raw["reviews"].copy())
    clean["products"]    = _clean_products(raw["products"].copy(), raw["categories"])
    clean["customers"]   = raw["customers"].copy()
    clean["sellers"]     = raw["sellers"].copy()
    clean["geolocation"] = raw["geolocation"].drop_duplicates("geolocation_zip_code_prefix").copy()
    clean["categories"]  = raw["categories"].copy()

    clean["fact_orders"] = _build_fact_orders(
        clean["orders"],
        clean["customers"],
        clean["order_items"],
        clean["payments"],
        clean["reviews"],
    )

    clean["weekly_agg"] = _build_weekly_agg(clean["fact_orders"])

    clean["seller_scorecard"] = _build_seller_scorecard(
        items     = clean["order_items"],
        orders    = clean["orders"],
        reviews   = clean["reviews"],
        sellers   = clean["sellers"],
        weights   = etl["health_score_weights"],
        min_orders= etl["min_sellers_for_scorecard"],
    )

    log.info(f"  Transform complete — {len(clean)} tables")
    return clean
