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


def test_query_timeout(db_path):
    db = ReadOnlyDB(db_path, timeout_s=0.2)
    endless = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r"
    with pytest.raises(TimeoutError):
        db.run_query(endless)
    # the connection still works afterwards
    assert db.run_query("SELECT COUNT(*) FROM Artist").rows == [(3,)]


def test_describe_table(db_path):
    text = ReadOnlyDB(db_path).describe_table("Album")
    assert "AlbumId INTEGER PRIMARY KEY" in text
    assert "FOREIGN KEY (ArtistId) REFERENCES Artist(ArtistId)" in text
    assert "Balls to the Wall" in text


def test_describe_unknown_table(db_path):
    with pytest.raises(ValueError, match="Unknown table"):
        ReadOnlyDB(db_path).describe_table("Nope")


def test_tool_specs_are_consistent():
    assert TOOL_NAMES == {"list_tables", "describe_table", "run_sql", "submit_answer"}
    for spec in TOOL_SPECS:
        assert spec["type"] == "function"
        assert spec["function"]["parameters"]["type"] == "object"


def test_toolbox_dispatch(db_path):
    tb = ToolBox(ReadOnlyDB(db_path))
    assert tb.call("list_tables", "{}").content == "Album, Artist"
    out = tb.call("run_sql", '{"query": "SELECT COUNT(*) AS n FROM Album"}')
    assert not out.is_error and "n\n3" in out.content
    final = tb.call("submit_answer", {"sql": "SELECT 1", "answer": "one"})
    assert final.final == {"sql": "SELECT 1", "answer": "one"}


@pytest.mark.parametrize(
    "name,args,expected",
    [
        ("drop_everything", "{}", "unknown tool"),
        ("run_sql", "{not json", "not valid JSON"),
        ("run_sql", "[1, 2]", "must be a JSON object"),
        ("run_sql", "{}", "missing argument"),
        ("run_sql", '{"query": "DELETE FROM Artist"}', "Only SELECT"),
        ("run_sql", '{"query": "SELECT nope FROM Artist"}', "no such column"),
        ("describe_table", '{"table": "Nope"}', "Unknown table"),
        ("submit_answer", '{"sql": "SELECT * FROM Missing", "answer": "x"}', "no such table"),
    ],
)
def test_toolbox_errors_are_returned_not_raised(db_path, name, args, expected):
    out = ToolBox(ReadOnlyDB(db_path)).call(name, args)
    assert out.is_error
    assert expected in out.content
    assert out.final is None


def test_toolbox_clips_long_output(db_path):
    tb = ToolBox(ReadOnlyDB(db_path), max_output_chars=30)
    out = tb.call("describe_table", '{"table": "Album"}')
    assert out.content.endswith("[output clipped]")

# Maintenance pass: 2026-04-01 11:26
