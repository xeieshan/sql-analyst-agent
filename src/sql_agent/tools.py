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
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A single SQLite SELECT statement"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Finish: submit the final SQL that answers the question, plus a one-sentence answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "Final SQLite SELECT statement"},
                    "answer": {"type": "string", "description": "One-sentence answer in plain English"},
                },
                "required": ["sql", "answer"],
            },
        },
    },
]

TOOL_NAMES = {spec["function"]["name"] for spec in TOOL_SPECS}


@dataclass
class ToolOutcome:
    content: str
    is_error: bool = False
    final: dict[str, str] | None = None


class ToolBox:
    def __init__(self, db: ReadOnlyDB, max_output_chars: int = 4000) -> None:
        self.db = db
        self.max_output_chars = max_output_chars
