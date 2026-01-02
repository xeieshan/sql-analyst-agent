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


@dataclass
class Step:
    index: int
    tool: str | None
    arguments: str | None
    output: str | None
    is_error: bool = False
    assistant_text: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


@dataclass
class Trajectory:
    question: str
    model: str
    steps: list[Step] = field(default_factory=list)
    final_sql: str | None = None
    final_answer: str | None = None
    stop_reason: str = "running"
    wall_time_s: float = 0.0

    @property
    def llm_calls(self) -> int:
        return len({s.index for s in self.steps})

    @property
    def tool_errors(self) -> int:
        return sum(s.is_error for s in self.steps)

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            llm_calls=self.llm_calls,
            tool_errors=self.tool_errors,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cost_usd=self.cost_usd,
        )
        return d


def _default_completion(**kwargs: Any) -> Any:
    import litellm

    return litellm.completion(**kwargs)


def _cost(resp: Any) -> float:
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception:  # noqa: BLE001 - local or unpriced models
        return 0.0


def _assistant_message(msg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            }
            for tc in msg.tool_calls
        ]
