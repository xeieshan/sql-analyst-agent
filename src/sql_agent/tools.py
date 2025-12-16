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

    def _clip(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + "\n... [output clipped]"

    def call(self, name: str, raw_args: str | dict | None) -> ToolOutcome:
        if name not in TOOL_NAMES:
            return ToolOutcome(f"Error: unknown tool {name!r}. Available: {sorted(TOOL_NAMES)}", is_error=True)
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else (raw_args or {})
        except json.JSONDecodeError as err:
            return ToolOutcome(f"Error: arguments are not valid JSON ({err})", is_error=True)
        if not isinstance(args, dict):
            return ToolOutcome("Error: arguments must be a JSON object", is_error=True)

        try:
            if name == "list_tables":
                return ToolOutcome(", ".join(self.db.list_tables()))
            if name == "describe_table":
                return ToolOutcome(self._clip(self.db.describe_table(str(args["table"]))))
            if name == "run_sql":
                return ToolOutcome(self._clip(self.db.run_query(str(args["query"])).to_text()))
            # submit_answer: check that the SQL runs before accepting it.
            sql = str(args["sql"])
            self.db.run_query(sql)
            return ToolOutcome("Answer submitted.", final={"sql": sql, "answer": str(args.get("answer", ""))})
        except KeyError as err:
            return ToolOutcome(f"Error: missing argument {err}", is_error=True)
        except (UnsafeQueryError, ValueError, TimeoutError, sqlite3.Error) as err:
            return ToolOutcome(f"Error: {err}", is_error=True)
