"""The agent loop: model -> tool calls -> tool results -> ... -> submit_answer.

Written directly on LiteLLM's OpenAI-compatible tool calling instead of an agent
framework, so every message, retry and stop condition is explicit and logged.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .tools import TOOL_SPECS, ToolBox

CompletionFn = Callable[..., Any]

SYSTEM_PROMPT = """You are a careful data analyst working with a SQLite database.
Answer the user's question by writing SQL.

How to work:
1. Inspect the schema with list_tables and describe_table before writing joins.
2. Test queries with run_sql and look at the result before trusting it.
3. Return exactly the columns the question asks for, in that order.
4. When the result answers the question, call submit_answer with the final SQL and a one-sentence answer.

Only read-only SELECT queries are allowed. Do not guess table or column names."""

NUDGE = "Please continue using the tools, and call submit_answer when you have the final query."

