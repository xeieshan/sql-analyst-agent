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


def download_chinook(dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(CHINOOK_URL, timeout=60) as resp:  # noqa: S310 - fixed https URL
        payload = resp.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CHINOOK_SHA256:
        raise RuntimeError(f"Checksum mismatch for Chinook download: {digest}")
    dest.write_bytes(payload)
    return dest


def validate_select(sql: str) -> str:
    """Return the normalised SQL if it is one read-only query, else raise UnsafeQueryError."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as err:
        raise UnsafeQueryError(f"Could not parse SQL: {err}") from err
    if len(statements) != 1:
        raise UnsafeQueryError(f"Expected exactly one statement, got {len(statements)}")
    stmt = statements[0]
    if not isinstance(stmt, exp.Query):
        raise UnsafeQueryError(f"Only SELECT queries are allowed, got {stmt.key.upper()}")
    for node in stmt.walk():
        if isinstance(node, _FORBIDDEN):
            raise UnsafeQueryError(f"Forbidden operation in query: {node.key.upper()}")
    return sql.strip().rstrip(";")


class ReadOnlyDB:
    def __init__(self, path: str | Path, timeout_s: float = 5.0, max_rows: int = 200) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Database not found: {self.path} (run `sql-agent download`)")
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self.conn = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True, check_same_thread=False)
        self.conn.execute("PRAGMA query_only = ON")

    def close(self) -> None:
        self.conn.close()

    def list_tables(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [r[0] for r in rows]

    def describe_table(self, table: str, sample_rows: int = 3) -> str:
        if table not in self.list_tables():
            raise ValueError(f"Unknown table: {table!r}. Tables: {', '.join(self.list_tables())}")
        cols = self.conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        fks = self.conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
        lines = [f"Table {table}:"]
        for _, name, ctype, notnull, _, pk in cols:
            flags = [f for f, on in (("PRIMARY KEY", pk), ("NOT NULL", notnull)) if on]
            lines.append(f"  {name} {ctype} {' '.join(flags)}".rstrip())
        for fk in fks:
            lines.append(f"  FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]})")
        sample = self.run_query(f'SELECT * FROM "{table}" LIMIT {int(sample_rows)}')
        lines.append("Sample rows:")
        lines.append(sample.to_text())
        return "\n".join(lines)

    def _execute(self, sql: str, limit: int | None) -> tuple[list[str], list[tuple], float]:
        sql = validate_select(sql)
        deadline = time.monotonic() + self.timeout_s
        self.conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        start = time.perf_counter()
        try:
            cur = self.conn.execute(sql)
            rows = cur.fetchall() if limit is None else cur.fetchmany(limit)
        except sqlite3.OperationalError as err:
            if "interrupted" in str(err):
                raise TimeoutError(f"Query exceeded {self.timeout_s}s") from err
            raise
        finally:
            self.conn.set_progress_handler(None, 0)
        columns = [d[0] for d in cur.description or []]
        return columns, [tuple(r) for r in rows], time.perf_counter() - start

    def run_query(self, sql: str) -> QueryResult:
        columns, rows, elapsed = self._execute(sql, limit=self.max_rows + 1)
        truncated = len(rows) > self.max_rows
        return QueryResult(columns, rows[: self.max_rows], truncated, elapsed)

    def run_unbounded(self, sql: str) -> list[tuple]:
        """Fetch every row (used for grading); still validated and time-limited."""
        return self._execute(sql, limit=None)[1]

# Maintenance pass: 2026-04-03 16:16
