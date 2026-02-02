"""Execution-accuracy evaluation.

A prediction is correct when its result set matches the gold query's result set:

* numbers are compared after rounding to 2 decimals (ints and floats compare equal),
* row order only matters when the question asks for an ordering (``order_matters``),
* extra columns are allowed: each gold column must map to a distinct predicted column
  with the same values (so ``SELECT name, id, total`` still matches ``SELECT name, total``).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .agent import SQLAgent, Trajectory
from .db import ReadOnlyDB
from .tools import ToolBox


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    gold_sql: str
    difficulty: str
    order_matters: bool


def load_questions(path: str | Path) -> list[Question]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.append(Question(r["id"], r["question"], r["gold_sql"], r["difficulty"], bool(r["order_matters"])))
    return out


def _norm(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int | float):
        return round(float(v), 2)
    if isinstance(v, str):
        return v.strip()
    return v


def _column(rows: list[tuple], j: int) -> list:
    return [r[j] for r in rows]


def _key(v: Any) -> tuple:
    return (v is None, str(type(v)), "" if v is None else v)


def results_match(gold: list[tuple], pred: list[tuple], order_matters: bool) -> bool:
    gold = [tuple(_norm(v) for v in r) for r in gold]
    pred = [tuple(_norm(v) for v in r) for r in pred]
    if len(gold) != len(pred):
        return False
    if not gold:
        return True
    n_gold, n_pred = len(gold[0]), len(pred[0])
    if n_pred < n_gold:
