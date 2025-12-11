"""Read-only access to a SQLite database, with the Chinook sample database as the default.

Safety is layered, because the SQL comes from a model:
1. sqlglot must parse the text as exactly one SELECT / WITH / UNION query.
2. The connection is opened with ``mode=ro`` and ``PRAGMA query_only = ON``.
3. A progress handler aborts queries that run longer than ``timeout_s``.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import sqlglot
from sqlglot import exp

CHINOOK_COMMIT = "7f67772503d71ba90f19283c38e93923addb43fa"
CHINOOK_URL = (
    f"https://raw.githubusercontent.com/lerocha/chinook-database/{CHINOOK_COMMIT}"
    "/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
)
CHINOOK_SHA256 = "7651ba378ac2fcd0dfc3c66fb101f7a7eed3ba39a612ec642b96e20702061f15"

_FORBIDDEN = (
    exp.Insert,
    exp.Update,
