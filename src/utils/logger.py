"""
utils/logger.py
───────────────
Centralised logger used by every module in the pipeline.
Creates a logs/ directory and writes to both stdout and a rotating log file.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Stage 1 complete")
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Return a configured logger that writes to stdout + file.

    Parameters
    ----------
    name     : module __name__
    log_dir  : directory for log files (default: project_root/logs)
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  [%(levelname)-8s]  %(name)s  |  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── stdout handler ──────────────────────────────────────────────────────
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # ── file handler ────────────────────────────────────────────────────────
    if log_dir is None:
        # Try to resolve project root
        here = Path(__file__).resolve().parent
        for parent in [here, *here.parents]:
            if (parent / "configs" / "config.yaml").exists():
                log_dir = str(parent / "logs")
                break
        else:
            log_dir = "logs"

    os.makedirs(log_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"pipeline_{today}.log")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
