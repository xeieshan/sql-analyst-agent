"""Tools exposed to the model, in OpenAI function-calling format (LiteLLM converts for other providers)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from .db import ReadOnlyDB, UnsafeQueryError

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all tables in the database.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Show a table's columns, types, keys, foreign keys and 3 sample rows.",
            "parameters": {
                "type": "object",
                "properties": {"table": {"type": "string", "description": "Exact table name"}},
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run one read-only SQLite SELECT query and see up to 20 rows. Use it to test queries.",
            "parameters": {
