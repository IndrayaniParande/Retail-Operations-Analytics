"""
utils/db.py
───────────
Database connection factory.
Returns a sqlite3 connection by default.
Swap get_connection() body for psycopg2 to move to PostgreSQL.

Usage:
    from src.utils.db import get_connection
    with get_connection(cfg) as conn:
        df = pd.read_sql("SELECT ...", conn)
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Generator

from src.utils.logger import get_logger

log = get_logger(__name__)


@contextmanager
def get_connection(cfg: dict) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields an open SQLite connection
    and commits/closes cleanly.

    Parameters
    ----------
    cfg : dict  Project config from load_config()

    Yields
    ------
    sqlite3.Connection
    """
    db_path = cfg["paths"]["database"]
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent read perf
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return True if table exists in the connected database."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cur.fetchone() is not None
