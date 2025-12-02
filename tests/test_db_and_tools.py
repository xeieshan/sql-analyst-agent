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
