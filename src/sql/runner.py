import argparse
import os

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.db import get_connection
from src.utils.logger import get_logger

log = get_logger(__name__)



QUERIES: dict[str, dict] = {

    "q1_rolling_revenue": {
        "description": "Rolling 30/90-day revenue — smooth trends, spot genuine growth",
        "sql": """
WITH daily_revenue AS (
    SELECT
        DATE(order_purchase_timestamp)   AS order_date,
        COUNT(order_id)                  AS daily_orders,
        ROUND(SUM(order_revenue), 2)     AS daily_revenue
    FROM   fact_orders
    WHERE  order_status = 'delivered'
      AND  order_purchase_timestamp IS NOT NULL
    GROUP  BY DATE(order_purchase_timestamp)
)
SELECT
    order_date,
    daily_orders,
    daily_revenue,
    ROUND(SUM(daily_revenue) OVER (
        ORDER BY order_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ), 2)  AS revenue_rolling_30d,
    ROUND(SUM(daily_revenue) OVER (
        ORDER BY order_date ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
    ), 2)  AS revenue_rolling_90d,
    ROUND(AVG(daily_orders) OVER (
        ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 1)  AS orders_ma7
FROM daily_revenue
ORDER BY order_date
""",
    },

    "q2_yoy_by_state": {
        "description": "Year-over-Year revenue by state — identify growth vs declining markets",
        "sql": """
WITH state_yearly AS (
    SELECT
        customer_state,
        CAST(strftime('%Y', order_purchase_timestamp) AS INTEGER) AS yr,
        COUNT(DISTINCT order_id)        AS total_orders,
        ROUND(SUM(order_revenue), 2)    AS total_revenue,
        ROUND(AVG(order_revenue), 2)    AS avg_order_value,
        ROUND(AVG(review_score), 3)     AS avg_review_score
    FROM   fact_orders
    WHERE  order_status = 'delivered'
    GROUP  BY customer_state, yr
)
SELECT
    curr.customer_state,
    curr.yr                              AS order_year,
    curr.total_orders,
    curr.total_revenue,
    curr.avg_order_value,
    curr.avg_review_score,
    prev.total_revenue                   AS prev_year_revenue,
    ROUND(
        (curr.total_revenue - prev.total_revenue)
        / NULLIF(prev.total_revenue, 0) * 100, 2
    )                                    AS revenue_yoy_pct,
    ROUND(
        (curr.total_orders - prev.total_orders)
        / NULLIF(CAST(prev.total_orders AS REAL), 0) * 100, 2
    )                                    AS orders_yoy_pct
FROM       state_yearly curr
LEFT JOIN  state_yearly prev
    ON  curr.customer_state = prev.customer_state
    AND curr.yr             = prev.yr + 1
WHERE curr.yr >= 2017
ORDER BY curr.yr DESC, revenue_yoy_pct DESC
""",
    },

    "q3_seller_ranking": {
        "description": "Seller performance ranking — revenue rank nationally and within state",
        "sql": """
WITH seller_metrics AS (
    SELECT
        s.seller_id,
        s.seller_state,
        COUNT(DISTINCT oi.order_id)             AS total_orders,
        ROUND(SUM(oi.price), 2)                 AS total_revenue,
        ROUND(AVG(oi.price), 2)                 AS avg_item_price,
        ROUND(AVG(r.review_score), 3)           AS avg_review_score,
        ROUND(AVG(o.is_late_delivery) * 100, 2) AS late_delivery_pct,
        COUNT(DISTINCT oi.product_id)           AS unique_products_sold
    FROM        sellers     s
    JOIN        order_items oi ON oi.seller_id = s.seller_id
    LEFT JOIN   orders      o  ON o.order_id   = oi.order_id
    LEFT JOIN   reviews     r  ON r.order_id   = oi.order_id
    WHERE o.order_status = 'delivered'
    GROUP  BY s.seller_id, s.seller_state
    HAVING total_orders >= 5
)
SELECT
    *,
    DENSE_RANK() OVER (ORDER BY total_revenue DESC)          AS revenue_rank_national,
    DENSE_RANK() OVER (PARTITION BY seller_state ORDER BY total_revenue DESC)
                                                             AS revenue_rank_state,
    CASE
        WHEN late_delivery_pct >= 30 THEN 'HIGH RISK'
        WHEN late_delivery_pct >= 15 THEN 'MEDIUM RISK'
        ELSE 'LOW RISK'
    END                                                      AS delivery_risk_tier,
    ROUND(
        (avg_review_score / 5.0 * 50)
        + ((100 - late_delivery_pct) / 100.0 * 30)
        + (CASE WHEN total_orders >= 50 THEN 20 ELSE total_orders * 20.0 / 50 END)
    , 1)                                                     AS seller_health_score
FROM seller_metrics
ORDER BY revenue_rank_national
LIMIT 200
""",
    },

    "q4_category_pareto": {
        "description": "Category Pareto — which categories drive 80% of GMV?",
        "sql": """
WITH cat_revenue AS (
    SELECT
        p.product_category_name_english     AS category,
        COUNT(DISTINCT oi.order_id)         AS total_orders,
        ROUND(SUM(oi.price), 2)             AS total_revenue,
        ROUND(AVG(oi.price), 2)             AS avg_price,
        ROUND(AVG(oi.freight_value / MAX(oi.price + oi.freight_value, 1)) * 100, 2)
                                            AS freight_cost_pct
    FROM        order_items oi
    JOIN        products    p  ON p.product_id = oi.product_id
    JOIN        orders      o  ON o.order_id   = oi.order_id
    WHERE o.order_status = 'delivered'
    GROUP  BY category
),
totals AS (SELECT SUM(total_revenue) AS grand_total FROM cat_revenue)
SELECT
    c.category,
    c.total_orders,
    c.total_revenue,
    c.avg_price,
    c.freight_cost_pct,
    ROUND(c.total_revenue / t.grand_total * 100, 2)          AS revenue_share_pct,
    ROUND(SUM(c.total_revenue) OVER (
        ORDER BY c.total_revenue DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) / t.grand_total * 100, 2)                              AS cumulative_share_pct,
    CASE WHEN SUM(c.total_revenue) OVER (
        ORDER BY c.total_revenue DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) / t.grand_total <= 0.80 THEN 'CORE (80%)' ELSE 'TAIL (20%)' END AS pareto_segment
FROM cat_revenue c, totals t
ORDER BY total_revenue DESC
""",
    },

    "q5_cohort_retention": {
        "description": "Monthly cohort retention — measure customer loyalty over 6 months",
        "sql": """
WITH first_order AS (
    SELECT
        c.customer_unique_id,
        MIN(DATE(o.order_purchase_timestamp, 'start of month')) AS cohort_month
    FROM   orders    o
    JOIN   customers c ON c.customer_id = o.customer_id
    WHERE  o.order_status = 'delivered'
    GROUP  BY c.customer_unique_id
),
all_orders AS (
    SELECT
        c.customer_unique_id,
        DATE(o.order_purchase_timestamp, 'start of month') AS order_month
    FROM   orders    o
    JOIN   customers c ON c.customer_id = o.customer_id
    WHERE  o.order_status = 'delivered'
),
cohort_data AS (
    SELECT
        f.cohort_month,
        CAST((julianday(a.order_month) - julianday(f.cohort_month)) / 30.5 AS INTEGER)
            AS period_number,
        COUNT(DISTINCT a.customer_unique_id) AS customers
    FROM all_orders  a
    JOIN first_order f ON f.customer_unique_id = a.customer_unique_id
    GROUP BY f.cohort_month, period_number
),
cohort_sizes AS (
    SELECT cohort_month, customers AS cohort_size
    FROM   cohort_data WHERE period_number = 0
)
SELECT
    d.cohort_month,
    s.cohort_size,
    d.period_number,
    d.customers                                        AS retained_customers,
    ROUND(d.customers * 100.0 / s.cohort_size, 2)    AS retention_rate_pct
FROM       cohort_data  d
JOIN       cohort_sizes s ON s.cohort_month = d.cohort_month
WHERE d.period_number BETWEEN 0 AND 6
ORDER BY d.cohort_month, d.period_number
""",
    },

    "q6_freight_margin_risk": {
        "description": "Freight margin by state — which states are logistics cost killers?",
        "sql": """
SELECT
    customer_state,
    COUNT(DISTINCT order_id)                                    AS total_orders,
    ROUND(SUM(order_revenue), 2)                                AS total_revenue,
    ROUND(SUM(order_freight), 2)                                AS total_freight_cost,
    ROUND(AVG(freight_pct), 2)                                  AS avg_freight_pct,
    ROUND(SUM(order_freight) / NULLIF(SUM(order_revenue),0)*100,2)
                                                                AS freight_revenue_ratio,
    ROUND(AVG(delivery_delay_days), 1)                          AS avg_delivery_delay_days,
    ROUND(AVG(review_score), 3)                                 AS avg_review_score,
    CASE
        WHEN SUM(order_freight)/NULLIF(SUM(order_revenue),0) > 0.20
        THEN 'MARGIN RISK'
        ELSE 'OK'
    END                                                         AS freight_status
FROM   fact_orders
WHERE  order_status = 'delivered'
GROUP  BY customer_state
ORDER  BY freight_revenue_ratio DESC
""",
    },

    "q7_payment_installments": {
        "description": "Payment behaviour by installment tier — credit exposure and AOV",
        "sql": """
SELECT
    CASE
        WHEN installments = 1             THEN '1x Cash'
        WHEN installments BETWEEN 2 AND 3 THEN '2-3x'
        WHEN installments BETWEEN 4 AND 6 THEN '4-6x'
        ELSE '7x+'
    END                                   AS installment_tier,
    payment_type,
    COUNT(*)                              AS order_count,
    ROUND(AVG(payment_value), 2)          AS avg_order_value,
    ROUND(SUM(payment_value), 2)          AS total_payment_volume,
    ROUND(AVG(review_score), 3)           AS avg_review_score
FROM       fact_orders
WHERE      order_status = 'delivered'
  AND      installments IS NOT NULL
GROUP BY   installment_tier, payment_type
ORDER BY   avg_order_value DESC
""",
    },

    "q8_delivery_vs_review": {
        "description": "Review score vs delivery timing — quantify the cost of late delivery",
        "sql": """
SELECT
    CASE
        WHEN delivery_delay_days <= -7 THEN '1. Very Early (7+ days)'
        WHEN delivery_delay_days <= -1 THEN '2. Early (1-6 days)'
        WHEN delivery_delay_days  = 0  THEN '3. On Time'
        WHEN delivery_delay_days <= 7  THEN '4. Late (1-7 days)'
        ELSE                                '5. Very Late (7+ days)'
    END                                     AS delivery_timing,
    COUNT(*)                                AS order_count,
    ROUND(AVG(review_score), 3)             AS avg_review_score,
    ROUND(
        SUM(CASE WHEN review_score >= 4 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                       AS positive_review_pct,
    ROUND(SUM(order_revenue), 2)            AS total_revenue_at_risk
FROM   fact_orders
WHERE  order_status = 'delivered'
  AND  delivery_delay_days IS NOT NULL
  AND  review_score IS NOT NULL
GROUP  BY delivery_timing
ORDER  BY delivery_timing
""",
    },
}




def run_query(name: str, cfg: dict = None) -> pd.DataFrame:
    """
    Execute a named query and return a DataFrame.

    Parameters
    ----------
    name : str  Query key from QUERIES dict
    cfg  : dict Project config
    """
    cfg = cfg or load_config()
    if name not in QUERIES:
        raise ValueError(f"Unknown query '{name}'. Available: {list(QUERIES.keys())}")

    q = QUERIES[name]
    log.info(f"  Running: {name}")
    log.info(f"  Purpose: {q['description']}")

    with get_connection(cfg) as conn:
        df = pd.read_sql(q["sql"], conn)

    log.info(f"  → {len(df):,} rows returned")

   
    proc_dir = cfg["paths"]["processed_data"]
    os.makedirs(proc_dir, exist_ok=True)
    out = os.path.join(proc_dir, f"{name}.csv")
    df.to_csv(out, index=False)
    log.info(f"  → Saved to {out}")

    return df


def run_all(cfg: dict = None) -> dict[str, pd.DataFrame]:
    """Run all defined queries and return results dict."""
    cfg = cfg or load_config()
    log.info("=" * 60)
    log.info("SQL QUERY RUNNER — All Queries")
    log.info("=" * 60)
    results = {}
    for name in QUERIES:
        try:
            results[name] = run_query(name, cfg)
        except Exception as e:
            log.error(f"  Query '{name}' failed: {e}")
    log.info("=" * 60)
    log.info(f"Complete — {len(results)}/{len(QUERIES)} queries succeeded")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Olist SQL Queries")
    parser.add_argument("--query", type=str, default=None,
                        help=f"Query name. Options: {list(QUERIES.keys())}")
    args = parser.parse_args()

    cfg = load_config()
    if args.query:
        run_query(args.query, cfg)
    else:
        run_all(cfg)
