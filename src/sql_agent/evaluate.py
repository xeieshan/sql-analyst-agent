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
        return False

    gold_cols = [sorted(_column(gold, j), key=_key) for j in range(n_gold)]
    pred_cols = [sorted(_column(pred, j), key=_key) for j in range(n_pred)]
    candidates = [[p for p in range(n_pred) if pred_cols[p] == gold_cols[g]] for g in range(n_gold)]

    def check(mapping: list[int]) -> bool:
        projected = [tuple(r[p] for p in mapping) for r in pred]
        if order_matters:
            return projected == gold
        return Counter(projected) == Counter(gold)

    def search(g: int, used: set[int], mapping: list[int]) -> bool:
        if g == n_gold:
            return check(mapping)
        for p in candidates[g]:
            if p not in used and search(g + 1, used | {p}, mapping + [p]):
                return True
        return False

    return search(0, set(), [])


def grade(db: ReadOnlyDB, q: Question, traj: Trajectory) -> dict:
    record = {
        "id": q.qid,
        "difficulty": q.difficulty,
        "stop_reason": traj.stop_reason,
        "final_sql": traj.final_sql,
        "correct": False,
        "error": None,
    }
    if traj.final_sql is None:
        record["error"] = f"no final SQL ({traj.stop_reason})"
        return record
    gold_rows = db.run_unbounded(q.gold_sql)
    try:
        pred_rows = db.run_unbounded(traj.final_sql)
    except Exception as err:  # noqa: BLE001 - any failure of the predicted SQL is a wrong answer
        record["error"] = f"final SQL failed: {err}"
        return record
    record["correct"] = results_match(gold_rows, pred_rows, q.order_matters)
    if not record["correct"]:
        record["error"] = "result mismatch"
    return record


def make_oracle(questions: Iterable[Question]) -> Callable[..., Any]:
    """A fake model that inspects the schema once, then submits the gold SQL.

    Running the harness with it must give 100% accuracy; anything less is a harness bug.
    """
    gold = {q.question: q.gold_sql for q in questions}

    def call(id_: str, name: str, args: dict) -> SimpleNamespace:
        return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=json.dumps(args)))

    def completion(**kwargs: Any) -> SimpleNamespace:
        messages = kwargs["messages"]
        question = messages[1]["content"]
        n_tool_msgs = sum(1 for m in messages if m["role"] == "tool")
        if n_tool_msgs == 0:
            calls = [call("c1", "list_tables", {})]
        else:
            calls = [call("c2", "submit_answer", {"sql": gold[question], "answer": "oracle"})]
        msg = SimpleNamespace(content=None, tool_calls=calls)
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    return completion


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def summarize(records: list[dict], trajectories: list[Trajectory]) -> dict:
    n = len(records)
    by_diff: dict[str, list[bool]] = {}
    for r in records:
        by_diff.setdefault(r["difficulty"], []).append(r["correct"])
    walls = [t.wall_time_s for t in trajectories]
    return {
        "n": n,
        "execution_accuracy": sum(r["correct"] for r in records) / n,
        "by_difficulty": {d: round(sum(v) / len(v), 3) for d, v in sorted(by_diff.items())},
        "submitted_rate": sum(t.final_sql is not None for t in trajectories) / n,
        "mean_llm_calls": statistics.mean(t.llm_calls for t in trajectories),
        "mean_tool_errors": statistics.mean(t.tool_errors for t in trajectories),
        "total_prompt_tokens": sum(t.prompt_tokens for t in trajectories),
        "total_completion_tokens": sum(t.completion_tokens for t in trajectories),
        "total_cost_usd": round(sum(t.cost_usd for t in trajectories), 4),
        "latency_p50_s": round(_pct(walls, 50), 2),
        "latency_p95_s": round(_pct(walls, 95), 2),
        "failures": dict(Counter(r["error"].split(":")[0] for r in records if r["error"])),
    }


def run_eval(
    db_path: str | Path,
    questions: list[Question],
    model: str,
    completion_fn: Callable[..., Any] | None = None,
    max_steps: int = 12,
    trajectory_path: str | Path | None = None,
) -> dict:
    db = ReadOnlyDB(db_path)
    agent = SQLAgent(ToolBox(db), model=model, max_steps=max_steps, completion_fn=completion_fn)
    records, trajs = [], []
    traj_file = None
    if trajectory_path:
        Path(trajectory_path).parent.mkdir(parents=True, exist_ok=True)
        traj_file = Path(trajectory_path).open("w", encoding="utf-8")
    try:
        for q in questions:
            traj = agent.run(q.question)
            rec = grade(db, q, traj)
            records.append(rec)
            trajs.append(traj)
            if traj_file:
                traj_file.write(json.dumps({"id": q.qid, "grade": rec, "trajectory": traj.to_dict()}) + "\n")
    finally:
        if traj_file:
            traj_file.close()
        db.close()
    return {"model": model, "summary": summarize(records, trajs), "per_question": records}

# Maintenance pass: 2026-02-26 18:51
