"""
anomaly/detector.py
────────────────────
Anomaly Detection Engine for weekly Olist sales data.

Three methods:
  1. Z-Score          — fast, explainable, assumes normality
  2. IQR Fences       — robust to skewed seasonal data
  3. Isolation Forest — ML, catches multivariate anomalies

Confidence levels based on method agreement:
  1/3 → Possible Anomaly
  2/3 → Likely Anomaly
  3/3 → Confirmed Anomaly

Usage:
    from src.anomaly.detector import run
    report_df = run(cfg)
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils.config_loader import load_config
from src.utils.db import get_connection
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Detection methods ─────────────────────────────────────────────────────────

def _zscore_detection(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Flag weeks where order_count or total_revenue deviate > threshold std."""
    for col in ["order_count", "total_revenue", "avg_review"]:
        filled = df[col].fillna(df[col].mean())
        df[f"z_{col}"] = stats.zscore(filled).round(3)

    df["zscore_anomaly"] = (
        (df["z_order_count"].abs()   > threshold) |
        (df["z_total_revenue"].abs() > threshold)
    ).astype(int)

    df["zscore_direction"] = np.where(
        df["z_order_count"] >  threshold, "SPIKE",
        np.where(df["z_order_count"] < -threshold, "DROP", "NORMAL"),
    )

    n = df["zscore_anomaly"].sum()
    log.info(f"  Z-Score  (±{threshold}σ)    → {n} anomalous weeks")
    return df


def _iqr_detection(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """IQR fence method — non-parametric, robust to skewed distributions."""
    for col in ["order_count", "total_revenue"]:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        df[f"iqr_flag_{col}"]  = ((df[col] < lower) | (df[col] > upper)).astype(int)
        df[f"iqr_lower_{col}"] = round(lower, 2)
        df[f"iqr_upper_{col}"] = round(upper, 2)

    df["iqr_anomaly"] = (
        (df["iqr_flag_order_count"] == 1) | (df["iqr_flag_total_revenue"] == 1)
    ).astype(int)

    n = df["iqr_anomaly"].sum()
    log.info(f"  IQR     (×{multiplier})       → {n} anomalous weeks")
    return df


def _isolation_forest_detection(
    df: pd.DataFrame,
    features: list,
    n_estimators: int,
    contamination: float,
    random_state: int,
) -> pd.DataFrame:
    """Isolation Forest — catches multivariate, non-linear anomalies."""
    X = df[features].fillna(df[features].mean())
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf    = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    preds  = clf.fit_predict(X_scaled)      # -1 = anomaly, 1 = normal
    scores = clf.score_samples(X_scaled)    # more negative = more anomalous

    df["iforest_anomaly"] = (preds == -1).astype(int)
    df["iforest_score"]   = scores.round(4)

    n = df["iforest_anomaly"].sum()
    log.info(f"  IsoForest ({contamination*100:.0f}% contam) → {n} anomalous weeks")
    return df


# ── Enrichment ────────────────────────────────────────────────────────────────

def _classify_anomaly(row: pd.Series) -> str:
    """Generate a plain-English business explanation for each flagged week."""
    reasons = []
    z_orders  = row.get("z_order_count",   0) or 0
    z_rev     = row.get("z_total_revenue", 0) or 0
    late      = row.get("late_count",  0) or 0
    orders    = row.get("order_count", 1) or 1
    review    = row.get("avg_review",  5) or 5

    if row.get("zscore_direction") == "SPIKE":
        reasons.append(f"Order volume spike (+{z_orders:.1f}σ above mean)")
    elif row.get("zscore_direction") == "DROP":
        reasons.append(f"Order volume drop ({z_orders:.1f}σ below mean)")

    if abs(z_rev) > 2.0:
        direction = "above" if z_rev > 0 else "below"
        reasons.append(f"Revenue {direction} baseline ({z_rev:.1f}σ)")

    if row.get("iqr_anomaly"):
        reasons.append("Outside IQR fence (non-parametric)")

    if row.get("iforest_anomaly"):
        reasons.append("Isolation Forest multivariate outlier")

    if late > orders * 0.4:
        reasons.append(f"High late delivery: {late}/{orders} orders ({late/orders*100:.0f}%)")

    if review < 3.0:
        reasons.append(f"Low avg review score ({review:.2f}/5.0)")

    return " | ".join(reasons) if reasons else "Normal week"


def _build_report(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all method flags, assign confidence, compute revenue impact."""
    df["consensus_anomaly"] = (
        df[["zscore_anomaly", "iqr_anomaly", "iforest_anomaly"]]
        .fillna(0).sum(axis=1).astype(int)
    )
    df["anomaly_confidence"] = df["consensus_anomaly"].map({
        0: "Normal",
        1: "Possible Anomaly",
        2: "Likely Anomaly",
        3: "Confirmed Anomaly",
    })

    df["business_explanation"] = df.apply(_classify_anomaly, axis=1)

    rolling_mean = df["total_revenue"].rolling(4, min_periods=1).mean().shift(1)
    df["revenue_vs_baseline"] = (df["total_revenue"] - rolling_mean).round(2)
    df["revenue_impact_pct"]  = (
        df["revenue_vs_baseline"] / rolling_mean * 100
    ).round(2)

    return df


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(cfg: dict = None) -> pd.DataFrame:
    """
    Load weekly_agg from DB, run all 3 detectors, save report.

    Parameters
    ----------
    cfg : dict  Project config

    Returns
    -------
    pd.DataFrame  Full anomaly report
    """
    cfg = cfg or load_config()
    acfg = cfg["anomaly"]

    log.info("=" * 60)
    log.info("ANOMALY DETECTION ENGINE")
    log.info("=" * 60)

    # Load weekly aggregation
    with get_connection(cfg) as conn:
        weekly = pd.read_sql("SELECT * FROM weekly_agg ORDER BY week_start", conn)

    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    weekly = (weekly[weekly["order_count"] >= acfg["min_orders_per_week"]]
              .reset_index(drop=True))

    log.info(f"  Loaded {len(weekly)} weeks "
             f"({weekly['week_start'].min().date()} → {weekly['week_start'].max().date()})")

    # Apply detectors
    weekly = _zscore_detection(weekly, acfg["zscore_threshold"])
    weekly = _iqr_detection(weekly, acfg["iqr_multiplier"])
    weekly = _isolation_forest_detection(
        weekly,
        features      = acfg["features"],
        n_estimators  = acfg["isolation_forest"]["n_estimators"],
        contamination = acfg["isolation_forest"]["contamination"],
        random_state  = acfg["isolation_forest"]["random_state"],
    )

    # Build enriched report
    report = _build_report(weekly)

    # Summary
    confirmed = (report["anomaly_confidence"] == "Confirmed Anomaly").sum()
    likely    = (report["anomaly_confidence"] == "Likely Anomaly").sum()
    possible  = (report["anomaly_confidence"] == "Possible Anomaly").sum()
    normal    = (report["anomaly_confidence"] == "Normal").sum()

    log.info(f"  Normal:             {normal}")
    log.info(f"  Possible Anomaly:   {possible}")
    log.info(f"  Likely Anomaly:     {likely}")
    log.info(f"  Confirmed Anomaly:  {confirmed}")

    top = (report[report["consensus_anomaly"] >= 2]
           .sort_values("revenue_impact_pct", key=abs, ascending=False)
           .head(5))
    for _, row in top.iterrows():
        impact = row.get("revenue_impact_pct", float("nan"))
        impact_str = f"{impact:+.1f}%" if pd.notna(impact) else "N/A"
        log.info(
            f"  ⚠  Week {str(row['week_start'])[:10]} | "
            f"Orders: {int(row['order_count']):>4} | "
            f"Revenue: R${row['total_revenue']:>10,.0f} | "
            f"Impact: {impact_str:>8} | "
            f"{row['anomaly_confidence']}"
        )

    # Persist
    proc_dir = cfg["paths"]["processed_data"]
    os.makedirs(proc_dir, exist_ok=True)
    out_path = os.path.join(proc_dir, "anomaly_report.csv")
    report.to_csv(out_path, index=False)
    log.info(f"  Saved → {out_path}")

    # Also save text summary
    _save_text_summary(report, cfg)

    return report


def _save_text_summary(report: pd.DataFrame, cfg: dict) -> None:
    """Write plain-text anomaly summary for logging/email."""
    import os
    root      = cfg["root"]
    rpt_dir   = os.path.join(root, "reports")
    os.makedirs(rpt_dir, exist_ok=True)
    txt_path  = os.path.join(rpt_dir, "anomaly_summary.txt")

    lines = [
        "=" * 60,
        "OLIST RETAIL — ANOMALY DETECTION SUMMARY",
        "=" * 60,
        f"Period : {report['week_start'].min().date()} → {report['week_start'].max().date()}",
        f"Weeks  : {len(report)}",
        "",
        "CONFIRMED ANOMALIES:",
        "-" * 60,
    ]
    confirmed = report[report["consensus_anomaly"] >= 2].sort_values(
        "revenue_impact_pct", key=abs, ascending=False
    )
    for _, r in confirmed.iterrows():
        impact = r.get("revenue_impact_pct", float("nan"))
        impact_str = f"{impact:+.1f}%" if pd.notna(impact) else "N/A"
        lines.append(
            f"  {str(r['week_start'])[:10]}  |  "
            f"Orders: {int(r['order_count']):>4}  |  "
            f"Revenue: R${r['total_revenue']:>10,.0f}  |  "
            f"Impact: {impact_str}"
        )
        lines.append(f"    → {r['business_explanation']}")

    lines += ["", "Methods: Z-Score ±2σ | IQR ×1.5 | Isolation Forest 5%", "=" * 60]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"  Text summary → {txt_path}")


if __name__ == "__main__":
    run()
