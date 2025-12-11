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
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Transaction,
)


class UnsafeQueryError(ValueError):
    """Raised when a query is not a single read-only SELECT."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    elapsed_s: float

    def to_text(self, max_rows: int = 20) -> str:
        if not self.columns:
            return "(no columns)"
        lines = [" | ".join(self.columns)]
        for row in self.rows[:max_rows]:
            lines.append(" | ".join("NULL" if v is None else str(v) for v in row))
        shown = min(len(self.rows), max_rows)
        more = len(self.rows) - shown
        footer = f"({len(self.rows)} rows{', truncated' if self.truncated else ''})"
        if more > 0:
            footer = f"({shown} of {len(self.rows)} rows shown{', more truncated' if self.truncated else ''})"
        return "\n".join(lines + [footer])
