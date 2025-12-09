import sqlite3

import pytest

from sql_agent.db import ReadOnlyDB, UnsafeQueryError, validate_select
from sql_agent.tools import TOOL_NAMES, TOOL_SPECS, ToolBox


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select * from Artist;",
        "WITH a AS (SELECT 1 AS x) SELECT x FROM a",
        "SELECT Name FROM Artist UNION SELECT Title FROM Album",
    ],
)
def test_accepts_read_only_queries(sql):
    assert validate_select(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM Artist",
        "DROP TABLE Artist",
        "UPDATE Artist SET Name = 'x'",
        "INSERT INTO Artist VALUES (9, 'x')",
        "PRAGMA table_info(Artist)",
        "ATTACH DATABASE 'x.db' AS y",
        "SELECT 1; DROP TABLE Artist",
        "CREATE TABLE t AS SELECT 1",
        "VACUUM",
        "this is not sql at all (",
    ],
)
def test_rejects_writes_and_multiple_statements(sql):
    with pytest.raises(UnsafeQueryError):
        validate_select(sql)


def test_connection_is_read_only_even_without_validation(db_path):
    db = ReadOnlyDB(db_path)
    with pytest.raises(sqlite3.OperationalError):
        db.conn.execute("DELETE FROM Artist")


def test_run_query_truncates(db_path):
    db = ReadOnlyDB(db_path, max_rows=2)
    res = db.run_query("SELECT Name FROM Artist ORDER BY ArtistId")
    assert res.columns == ["Name"]
    assert res.rows == [("AC/DC",), ("Accept",)]
    assert res.truncated
    assert "truncated" in res.to_text()


