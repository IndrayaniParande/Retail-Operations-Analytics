"""
utils/config_loader.py
──────────────────────
Central config loader. Reads configs/config.yaml and resolves all
relative paths to absolute paths from the project root.

Usage:
    from src.utils.config_loader import load_config
    cfg = load_config()
    db_path = cfg["paths"]["database"]
"""

import os
import yaml
from pathlib import Path


def get_project_root() -> Path:
    """Walk up from this file until we find configs/config.yaml."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "configs" / "config.yaml").exists():
            return parent
    raise FileNotFoundError(
        "Could not locate project root (no configs/config.yaml found). "
        "Make sure you run commands from the project root."
    )


def load_config(config_path: str = None) -> dict:
    """
    Load and return the project config with all paths resolved
    to absolute paths relative to the project root.

    Parameters
    ----------
    config_path : str, optional
        Override path to a config file. Defaults to configs/config.yaml.

    Returns
    -------
    dict
        Full config dictionary with 'root' key added.
    """
    root = get_project_root()

    if config_path is None:
        config_path = root / "configs" / "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve all path values to absolute
    cfg["root"] = str(root)
    for key, rel_path in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(root / rel_path)

    return cfg
