import argparse
import os
from datetime import datetime, timedelta

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src.utils.config_loader import load_config
from src.utils.db import get_connection
from src.utils.logger import get_logger

log = get_logger(__name__)


#Formating

def _fmt(v, decimals: int = 0) -> str:
    if pd.isna(v):
        return "N/A"
    return f"{v:,.{decimals}f}"


def _delta(curr: float, prev: float) -> tuple[str, str]:
    """Return (formatted_pct_string, direction) for WoW comparisons."""
    if prev == 0 or pd.isna(prev):
        return "N/A", "neutral"
    pct  = (curr - prev) / abs(prev) * 100
    sign = "+" if pct >= 0 else ""
    direction = "up" if pct >= 1 else "down" if pct <= -1 else "neutral"
    return f"{sign}{pct:.1f}%", direction


# Data fetchers 

def _fetch_kpis(conn, ws: str, we: str, ps: str, pe: str) -> dict:
    curr = pd.read_sql(f"""
        SELECT order_id, order_revenue, order_freight, is_late_delivery, review_score
        FROM fact_orders
        WHERE order_purchase_timestamp >= '{ws}' AND order_purchase_timestamp <= '{we}'
          AND order_status = 'delivered'
    """, conn)

    prev = pd.read_sql(f"""
        SELECT order_id, order_revenue FROM fact_orders
        WHERE order_purchase_timestamp >= '{ps}' AND order_purchase_timestamp <= '{pe}'
          AND order_status = 'delivered'
    """, conn)

    curr_orders  = len(curr)
    curr_revenue = curr["order_revenue"].sum()
    prev_orders  = len(prev)
    prev_revenue = prev["order_revenue"].sum()
    late_n       = int(pd.to_numeric(curr["is_late_delivery"], errors="coerce").fillna(0).sum())
    late_pct     = round(late_n / max(curr_orders, 1) * 100, 1)
    review_avg   = round(float(curr["review_score"].mean() or 0), 2)
    review_cnt   = int(curr["review_score"].notna().sum())

    orders_delta,  orders_dir  = _delta(curr_orders,  prev_orders)
    revenue_delta, revenue_dir = _delta(curr_revenue, prev_revenue)

    return {
        "total_orders":      f"{curr_orders:,}",
        "total_revenue":     _fmt(curr_revenue),
        "review_avg":        review_avg,
        "review_count":      f"{review_cnt:,}",
        "late_pct":          late_pct,
        "late_orders":       late_n,
        "orders_delta":      orders_delta,
        "orders_delta_dir":  orders_dir,
        "revenue_delta":     revenue_delta,
        "revenue_delta_dir": revenue_dir,
    }


def _fetch_top_states(conn, ws: str, we: str, n: int) -> list:
    df = pd.read_sql(f"""
        SELECT customer_state,
               COUNT(order_id)        AS orders,
               SUM(order_revenue)     AS revenue
        FROM fact_orders
        WHERE order_purchase_timestamp >= '{ws}' AND order_purchase_timestamp <= '{we}'
          AND order_status = 'delivered'
        GROUP BY customer_state ORDER BY revenue DESC LIMIT {n}
    """, conn)
    total = df["revenue"].sum()
    return [
        {
            "state":     r["customer_state"],
            "orders":    f"{int(r['orders']):,}",
            "revenue":   _fmt(r["revenue"]),
            "share_pct": round(r["revenue"] / max(total, 1) * 100, 1),
        }
        for _, r in df.iterrows()
    ]


def _fetch_payment_mix(conn, ws: str, we: str) -> list:
    df = pd.read_sql(f"""
        SELECT p.payment_type,
               COUNT(*)                    AS orders,
               SUM(p.payment_value)        AS volume,
               AVG(p.payment_installments) AS avg_inst
        FROM payments p
        JOIN fact_orders f ON f.order_id = p.order_id
        WHERE f.order_purchase_timestamp >= '{ws}' AND f.order_purchase_timestamp <= '{we}'
          AND f.order_status = 'delivered'
        GROUP BY p.payment_type ORDER BY orders DESC
    """, conn)
    return [
        {
            "type":     r["payment_type"].replace("_"," ").title(),
            "orders":   f"{int(r['orders']):,}",
            "volume":   _fmt(r["volume"]),
            "avg_inst": _fmt(r["avg_inst"], 1),
        }
        for _, r in df.iterrows()
    ]


def _fetch_top_categories(conn, ws: str, we: str, ps: str, pe: str, n: int) -> list:
    curr = pd.read_sql(f"""
        SELECT p.product_category_name_english AS category,
               COUNT(DISTINCT oi.order_id)     AS orders,
               SUM(oi.price)                   AS revenue,
               AVG(oi.price)                   AS avg_price,
               AVG(oi.freight_value / MAX(oi.price + oi.freight_value, 1)) * 100 AS freight_pct
        FROM order_items oi
        JOIN products    p  ON p.product_id = oi.product_id
        JOIN fact_orders f  ON f.order_id   = oi.order_id
        WHERE f.order_purchase_timestamp >= '{ws}' AND f.order_purchase_timestamp <= '{we}'
          AND f.order_status = 'delivered'
        GROUP BY category ORDER BY revenue DESC LIMIT {n}
    """, conn)

    prev = pd.read_sql(f"""
        SELECT p.product_category_name_english AS category,
               SUM(oi.price) AS prev_rev
        FROM order_items oi
        JOIN products    p  ON p.product_id = oi.product_id
        JOIN fact_orders f  ON f.order_id   = oi.order_id
        WHERE f.order_purchase_timestamp >= '{ps}' AND f.order_purchase_timestamp <= '{pe}'
          AND f.order_status = 'delivered'
        GROUP BY category
    """, conn)

    merged = curr.merge(prev, on="category", how="left")
    results = []
    for _, r in merged.iterrows():
        rev  = r["revenue"] or 0
        prev_rev = r.get("prev_rev") or 0
        if prev_rev > 0:
            trend = "up" if rev > prev_rev * 1.05 else "down" if rev < prev_rev * 0.95 else "stable"
        else:
            trend = "stable"
        results.append({
            "category":      (r["category"] or "other").replace("_"," ").title(),
            "orders":        f"{int(r['orders']):,}",
            "revenue":       _fmt(rev),
            "avg_price":     _fmt(r["avg_price"], 2),
            "freight_pct":   round(r["freight_pct"] or 0, 1),
            "revenue_trend": trend,
        })
    return results


def _fetch_at_risk_sellers(conn, late_threshold: float, min_orders: int) -> list:
    df = pd.read_sql(f"""
        SELECT seller_id, seller_state, total_orders, total_revenue,
               avg_review_score, late_delivery_pct
        FROM seller_scorecard
        WHERE late_delivery_pct > {late_threshold} AND total_orders >= {min_orders}
        ORDER BY late_delivery_pct DESC LIMIT 8
    """, conn)
    return [
        {
            "seller_id": r["seller_id"],
            "state":     r["seller_state"],
            "orders":    f"{int(r['total_orders']):,}",
            "revenue":   _fmt(r["total_revenue"]),
            "review":    round(r["avg_review_score"] or 0, 2),
            "late_pct":  round(r["late_delivery_pct"] or 0, 1),
        }
        for _, r in df.iterrows()
    ]


def _fetch_weekly_trend(conn, n_weeks: int) -> list:
    df = pd.read_sql(f"""
        SELECT week_start, order_count, total_revenue, avg_ticket, avg_review, late_count
        FROM weekly_agg ORDER BY week_start DESC LIMIT {n_weeks + 1}
    """, conn)
    df = df.sort_values("week_start").reset_index(drop=True)
    results = []
    for i, r in df.tail(n_weeks).iterrows():
        prev_rev = df.loc[i-1, "total_revenue"] if i > 0 else r["total_revenue"]
        wow_str, _ = _delta(r["total_revenue"], prev_rev)
        try:
            wow_num = float(wow_str.replace("+","").replace("%",""))
        except Exception:
            wow_num = 0.0
        late_pct = round(r["late_count"] / max(r["order_count"], 1) * 100, 1)
        results.append({
            "week":       str(r["week_start"])[:10],
            "orders":     f"{int(r['order_count']):,}",
            "revenue":    _fmt(r["total_revenue"]),
            "avg_ticket": _fmt(r["avg_ticket"], 2),
            "avg_review": round(r["avg_review"] or 0, 2),
            "late_pct":   late_pct,
            "wow_pct":    wow_num,
        })
    return results


def _fetch_anomalies(anom_csv: str, week_start: pd.Timestamp) -> list:
    if not os.path.exists(anom_csv):
        return []
    df = pd.read_csv(anom_csv)
    df["week_start"] = pd.to_datetime(df["week_start"])
    cutoff = (week_start - timedelta(days=14)).to_pydatetime()
    relevant = df[(df["week_start"] >= cutoff) & (df["consensus_anomaly"] >= 2)]
    results = []
    for _, a in relevant.iterrows():
        impact_val = a.get("revenue_impact_pct", float("nan"))
        impact_str = f"{impact_val:+.1f}" if pd.notna(impact_val) else "N/A"
        results.append({
            "week":        str(a["week_start"])[:10],
            "confidence":  a["anomaly_confidence"],
            "orders":      f"{int(a['order_count']):,}",
            "revenue":     _fmt(a["total_revenue"]),
            "impact":      impact_str,
            "explanation": a.get("business_explanation", ""),
        })
    return results




def build_report(week_str: str = None, cfg: dict = None) -> str:
    """
    Generate HTML report for the given week.

    Parameters
    ----------
    week_str : str  Week start date YYYY-MM-DD (latest week if None)
    cfg      : dict Project config

    Returns
    -------
    str  Path to the generated HTML file
    """
    cfg = cfg or load_config()
    rcfg = cfg["reports"]
    root = cfg["root"]
    tmpl_dir = cfg["paths"]["templates"]
    rpt_dir  = os.path.join(root, "reports")
    anom_csv = os.path.join(cfg["paths"]["processed_data"], "anomaly_report.csv")
    os.makedirs(rpt_dir, exist_ok=True)

    with get_connection(cfg) as conn:
        if week_str is None:
            row = pd.read_sql(
                "SELECT MAX(order_purchase_timestamp) AS mx FROM fact_orders", conn
            ).iloc[0]
            max_dt     = pd.to_datetime(row["mx"])
            week_start = max_dt - timedelta(days=max_dt.weekday() + 7)
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            week_start = pd.to_datetime(week_str)

        week_end   = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        prev_start = week_start - timedelta(days=7)
        prev_end   = week_start - timedelta(seconds=1)

        ws = week_start.strftime("%Y-%m-%d")
        we = week_end.strftime("%Y-%m-%d %H:%M:%S")
        ps = prev_start.strftime("%Y-%m-%d")
        pe = prev_end.strftime("%Y-%m-%d %H:%M:%S")

        log.info(f"  Building report for week: {ws}")

        data = {
            "report_week":       ws,
            "generated_at":      datetime.now().strftime("%Y-%m-%d %H:%M"),
            "data_through":      we[:10],
            "kpis":              _fetch_kpis(conn, ws, we, ps, pe),
            "top_states":        _fetch_top_states(conn, ws, we, rcfg["top_states_count"]),
            "payment_mix":       _fetch_payment_mix(conn, ws, we),
            "top_categories":    _fetch_top_categories(conn, ws, we, ps, pe, rcfg["top_categories_count"]),
            "at_risk_sellers":   _fetch_at_risk_sellers(conn, rcfg["at_risk_late_pct_threshold"], rcfg["at_risk_min_orders"]),
            "weekly_trend":      _fetch_weekly_trend(conn, rcfg["weekly_trend_weeks"]),
            "anomalies":         _fetch_anomalies(anom_csv, week_start),
        }
        data["has_anomalies"] = len(data["anomalies"]) > 0
        data["anomaly_count"] = len(data["anomalies"])

    env      = Environment(loader=FileSystemLoader(tmpl_dir))
    template = env.get_template("weekly_report.html")
    html     = template.render(**data)

    out_path = os.path.join(rpt_dir, f"weekly_report_{ws}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    log.info(f"  ✓ Report → {out_path}")
    return out_path


def run(cfg: dict = None) -> None:
    """Generate reports for all demo weeks defined in config."""
    cfg = cfg or load_config()
    demo_weeks = cfg["reports"]["demo_weeks"]

    log.info("=" * 60)
    log.info("REPORT GENERATOR")
    log.info("=" * 60)

    for week in demo_weeks:
        try:
            build_report(week, cfg)
        except Exception as e:
            log.warning(f"  Could not generate week {week}: {e}")

    
    try:
        build_report(None, cfg)
    except Exception as e:
        log.warning(f"  Could not generate latest week: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Olist Weekly Report")
    parser.add_argument("--week", type=str, default=None,
                        help="Week start date YYYY-MM-DD (default: latest week)")
    args = parser.parse_args()

    cfg = load_config()
    if args.week:
        build_report(args.week, cfg)
    else:
        run(cfg)
