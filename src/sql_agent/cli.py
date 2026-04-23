"""Command line interface: ``sql-agent <command> --help``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import SQLAgent
from .db import ReadOnlyDB, download_chinook
from .evaluate import load_questions, make_oracle, run_eval
from .tools import ToolBox

DEFAULT_DB = "data/chinook.sqlite"
DEFAULT_QUESTIONS = "data/questions.jsonl"


def _safe_name(model: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in model)


def cmd_download(args: argparse.Namespace) -> None:
    print(f"Saved {download_chinook(args.dest)}")


def cmd_ask(args: argparse.Namespace) -> None:
    db = ReadOnlyDB(args.db)
    traj = SQLAgent(ToolBox(db), model=args.model, max_steps=args.max_steps).run(args.question)
    if args.verbose:
        for s in traj.steps:
            if s.tool:
                print(f"[{s.index}] {s.tool}({s.arguments})\n    -> {(s.output or '')[:300]}")
            elif s.assistant_text:
                print(f"[{s.index}] model: {s.assistant_text[:300]}")
    print(f"\nSQL: {traj.final_sql}\nAnswer: {traj.final_answer}\nStop: {traj.stop_reason}")
    if traj.final_sql:
        print(db.run_query(traj.final_sql).to_text())


def cmd_eval(args: argparse.Namespace) -> None:
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    completion_fn = make_oracle(questions) if args.model == "oracle" else None
    out_dir = Path(args.out_dir)
    name = _safe_name(args.model)
    result = run_eval(
        args.db,
        questions,
        model=args.model,
        completion_fn=completion_fn,
        max_steps=args.max_steps,
        trajectory_path=out_dir / "trajectories" / f"{name}.jsonl",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


def cmd_check_gold(args: argparse.Namespace) -> None:
    db = ReadOnlyDB(args.db)
    for q in load_questions(args.questions):
        rows = db.run_unbounded(q.gold_sql)
        preview = rows[:3]
        print(f"{q.qid:>4} {q.difficulty:<6} rows={len(rows):<4} {preview}")


def cmd_report(args: argparse.Namespace) -> None:
    rows = [
        "| Model | Exec. accuracy | Easy | Medium | Hard | LLM calls / q | Tool errors / q | Cost (USD) "
        "| p50 latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for path in sorted(Path(args.results_dir).glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        s = r["summary"]
        d = s["by_difficulty"]
        rows.append(
            f"| {r['model']} | {s['execution_accuracy']:.3f} | {d.get('easy', 0):.2f} | {d.get('medium', 0):.2f} | "
            f"{d.get('hard', 0):.2f} | {s['mean_llm_calls']:.1f} | {s['mean_tool_errors']:.2f} | "
            f"{s['total_cost_usd']:.4f} | {s['latency_p50_s']:.2f} |"
        )
    table = "\n".join(rows)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sql-agent", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("download", help="download the pinned Chinook SQLite database")
    s.add_argument("--dest", default=DEFAULT_DB)
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("ask", help="ask one question")
    s.add_argument("question")
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--db", default=DEFAULT_DB)
    s.add_argument("--max-steps", type=int, default=12)
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("eval", help="run the benchmark; --model oracle validates the harness")
    s.add_argument("--model", default="gpt-4o-mini")
