"""
main.py
────────
Single entry point for the entire Olist Analytics Pipeline.

Stages (all or individual):
    generate  → Synthetic dataset (skip if using real Olist CSVs)
    etl       → Extract → Transform → Load
    anomaly   → Z-Score + IQR + Isolation Forest detection
    sql       → Run all 8 advanced SQL queries
    report    → Generate Jinja2 HTML weekly reports

Usage:
    # Full pipeline (no data generation)
    python main.py

    # Full pipeline + generate synthetic data
    python main.py --generate

    # Run specific stages only
    python main.py --stages etl anomaly

    # Generate report for a specific week
    python main.py --stages report --week 2017-11-20
"""

import argparse
import time

from src.anomaly.detector import run as run_anomaly
from olist_project.src.etl.data import run as generate_data
from src.etl.pipeline import run_pipeline
from src.reports.builder import build_report, run as run_reports
from src.sql.runner import run_all as run_sql
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)

ALL_STAGES = ["generate", "etl", "anomaly", "sql", "report"]


def main():
    parser = argparse.ArgumentParser(
        description="Olist E-Commerce Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # full pipeline
  python main.py --generate                   # include data generation
  python main.py --stages etl                 # ETL only
  python main.py --stages anomaly sql         # anomaly + SQL only
  python main.py --stages report --week 2017-11-20
        """,
    )
    parser.add_argument(
        "--stages", nargs="+", choices=ALL_STAGES, default=None,
        help="Stages to run (default: etl anomaly sql report)",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate synthetic raw data before ETL",
    )
    parser.add_argument(
        "--week", type=str, default=None,
        help="Week for report (YYYY-MM-DD). Only used with --stages report",
    )
    args = parser.parse_args()

    cfg    = load_config()
    stages = args.stages or (["generate", "etl", "anomaly", "sql", "report"]
                              if args.generate else ["etl", "anomaly", "sql", "report"])

    t_start = time.time()
    log.info("╔" + "═" * 58 + "╗")
    log.info("║   OLIST ANALYTICS PIPELINE — STARTING                  ║")
    log.info("╚" + "═" * 58 + "╝")
    log.info(f"Stages: {stages}")

    if "generate" in stages or args.generate:
        log.info("\n[1/5] GENERATE DATA")
        generate_data(cfg)

    if "etl" in stages:
        log.info("\n[2/5] ETL PIPELINE")
        run_pipeline(cfg, generate=False)

    if "anomaly" in stages:
        log.info("\n[3/5] ANOMALY DETECTION")
        run_anomaly(cfg)

    if "sql" in stages:
        log.info("\n[4/5] SQL QUERIES")
        run_sql(cfg)

    if "report" in stages:
        log.info("\n[5/5] REPORT GENERATION")
        if args.week:
            build_report(args.week, cfg)
        else:
            run_reports(cfg)

    elapsed = time.time() - t_start
    log.info("\n╔" + "═" * 58 + "╗")
    log.info(f"║   PIPELINE COMPLETE  —  {elapsed:.1f}s" + " " * (33 - len(f"{elapsed:.1f}")) + "║")
    log.info("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
