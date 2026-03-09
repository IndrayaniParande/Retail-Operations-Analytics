import os
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


def extract(cfg: dict = None) -> dict[str, pd.DataFrame]:
    """
    Load all 9 Olist CSV files from data/raw/.

    Parameters
    ----------
    cfg : dict  Project config from load_config()

    Returns
    -------
    dict mapping logical table name → DataFrame
    """
    cfg  = cfg or load_config()
    raw_dir   = cfg["paths"]["raw_data"]
    file_map  = cfg["data"]["files"]

    log.info("─" * 50)
    log.info("STAGE 1 — EXTRACT")
    log.info("─" * 50)

    raw: dict[str, pd.DataFrame] = {}
    missing = []

    for key, fname in file_map.items():
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            missing.append(path)
            log.error(f"  ✗ Missing: {path}")
            continue

        df = pd.read_csv(path, low_memory=False)
        raw[key] = df
        log.info(f"  ✓ {key:<15} → {len(df):>8,} rows  |  {df.shape[1]} cols")

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} raw file(s) not found.\n"
            "Run: python -m src.etl.generate_data  (for synthetic data)\n"
            "Or place real Olist CSVs in data/raw/ and retry.\n"
            f"Missing: {missing}"
        )

    log.info(f"  Extract complete — {len(raw)} tables loaded")
    return raw
