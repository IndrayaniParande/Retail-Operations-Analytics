import os
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.db import get_connection
from src.utils.logger import get_logger

log = get_logger(__name__)

# Tables to export (order matters for foreign key consistency)
TABLE_ORDER = [
    "customers",
    "sellers",
    "products",
    "categories",
    "geolocation",
    "orders",
    "order_items",
    "payments",
    "reviews",
    "fact_orders",
    "weekly_agg",
    "seller_scorecard",
]


def _df_to_sql(df: pd.DataFrame, name: str, conn) -> None:
    """Write a DataFrame to SQLite, converting timestamps to strings."""
    out = df.copy()
    for col in out.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        out[col] = out[col].astype(str)
    # Convert pandas nullable int types to standard int/float for SQLite
    for col in out.columns:
        if hasattr(out[col], "dtype") and str(out[col].dtype) in ("Int8", "Int32", "Int64"):
            out[col] = out[col].astype("float64")
    out.to_sql(name, conn, if_exists="replace", index=False)


def load(clean: dict[str, pd.DataFrame], cfg: dict = None) -> None:
    """
    Load all clean tables to SQLite and write processed CSVs.

    Parameters
    ----------
    clean : dict  Output from transform()
    cfg   : dict  Project config
    """
    cfg      = cfg or load_config()
    proc_dir = cfg["paths"]["processed_data"]
    indexes  = cfg["etl"]["indexes"]
    os.makedirs(proc_dir, exist_ok=True)

    log.info("─" * 50)
    log.info("STAGE 3 — LOAD")
    log.info("─" * 50)

    # ── SQLite ────────────────────────────────────────────────────────────────
    with get_connection(cfg) as conn:
        log.info(f"  Connected → {cfg['paths']['database']}")

        for tname in TABLE_ORDER:
            if tname not in clean:
                log.warning(f"  Table '{tname}' not found in clean dict — skipping")
                continue
            _df_to_sql(clean[tname], tname, conn)
            log.info(f"  ✓ {tname:<20} → {len(clean[tname]):>8,} rows")

        # Create indexes
        for idx_sql in indexes:
            try:
                conn.execute(idx_sql)
            except Exception as e:
                log.warning(f"  Index skipped: {e}")

        log.info(f"  ✓ {len(indexes)} indexes applied")

    # ── Processed CSVs ────────────────────────────────────────────────────────
    log.info(f"  Writing CSVs → {proc_dir}")
    for tname in TABLE_ORDER:
        if tname not in clean:
            continue
        path = os.path.join(proc_dir, f"{tname}.csv")
        clean[tname].to_csv(path, index=False)
        log.info(f"  ✓ {tname}.csv")

    log.info("─" * 50)
    log.info("LOAD complete")
    log.info("─" * 50)
