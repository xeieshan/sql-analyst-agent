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
