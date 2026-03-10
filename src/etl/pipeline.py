import argparse
import time

from src.etl.extract import extract
from src.etl.load import load
from src.etl.transform import transform
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


def run_pipeline(cfg: dict = None, generate: bool = False) -> dict:
    cfg = cfg or load_config()
    t0  = time.time()

    log.info("=" * 60)
    log.info("OLIST ETL PIPELINE — START")
    log.info("=" * 60)

    if generate:
        log.info("Generating synthetic dataset...")
        generate_data(cfg)

    raw   = extract(cfg)
    clean = transform(raw, cfg)
    load(clean, cfg)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"PIPELINE COMPLETE  —  {elapsed:.1f}s")
    log.info("=" * 60)

    return clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Olist ETL Pipeline")
    parser.add_argument(
        "--generate", action="store_true",
        help="Regenerate synthetic raw data before running ETL",
    )
    args = parser.parse_args()
    run_pipeline(generate=args.generate)
