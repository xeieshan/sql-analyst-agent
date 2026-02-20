import json
from pathlib import Path

import pytest
from helpers import response, tool_call

from sql_agent.db import ReadOnlyDB
from sql_agent.evaluate import Question, load_questions, make_oracle, results_match, run_eval

CHINOOK = Path("data/chinook.sqlite")


@pytest.mark.parametrize(
    "gold,pred,order,expected",
    [
        ([(1,)], [(1.0,)], False, True),
        ([(2328.6,)], [(2328.5999999999,)], False, True),
        ([("a", 1), ("b", 2)], [("b", 2), ("a", 1)], False, True),
        ([("a", 1), ("b", 2)], [("b", 2), ("a", 1)], True, False),
        ([("a", 1)], [(1, "a")], False, True),  # column order is free
        ([("a", 1)], [(99, "a", 1)], False, True),  # extra columns allowed
        ([("a", 1)], [("a",)], False, False),  # missing column
        ([("a", 1), ("a", 1)], [("a", 1)], False, False),  # duplicates matter
        ([("a", 1)], [("a", 2)], False, False),
        ([], [], False, True),
        ([(None,)], [(None,)], False, True),
        ([("x", 1), ("y", 1)], [(1, "y"), (1, "x")], True, False),
    ],
)
def test_results_match(gold, pred, order, expected):
    assert results_match(gold, pred, order) is expected


def test_columns_with_identical_values_are_disambiguated():
    gold = [(1, 2), (2, 1)]
    pred = [(2, 1), (1, 2)]  # each column alone matches either gold column
    assert results_match(gold, pred, order_matters=True)


def test_oracle_scores_100_percent_on_fixture(db_path, tmp_path):
    qs = [
        Question("q1", "How many artists?", "SELECT COUNT(*) FROM Artist", "easy", False),
        Question("q2", "Albums per artist", "SELECT ArtistId, COUNT(*) FROM Album GROUP BY ArtistId", "medium", False),
    ]
    res = run_eval(db_path, qs, model="oracle", completion_fn=make_oracle(qs), trajectory_path=tmp_path / "t.jsonl")
    assert res["summary"]["execution_accuracy"] == 1.0
    assert res["summary"]["mean_llm_calls"] == 2
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["grade"]["correct"] is True

