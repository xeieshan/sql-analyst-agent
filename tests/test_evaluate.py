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
