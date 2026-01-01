from helpers import response, tool_call

from sql_agent.agent import SQLAgent
from sql_agent.db import ReadOnlyDB
from sql_agent.tools import ToolBox


def make_agent(db_path, fn, max_steps=6):
    return SQLAgent(ToolBox(ReadOnlyDB(db_path)), model="test-model", max_steps=max_steps, completion_fn=fn)


def test_happy_path(db_path, scripted):
    fn = scripted(
        [
            response(tool_calls=[tool_call("1", "list_tables")]),
            response(tool_calls=[tool_call("2", "run_sql", query="SELECT COUNT(*) FROM Album")]),
            response(tool_calls=[tool_call("3", "submit_answer", sql="SELECT COUNT(*) FROM Album", answer="3")]),
        ]
    )
    traj = make_agent(db_path, fn).run("How many albums?")
    assert traj.stop_reason == "submitted"
    assert traj.final_sql == "SELECT COUNT(*) FROM Album"
    assert traj.llm_calls == 3
    assert traj.tool_errors == 0
    assert traj.prompt_tokens == 300
    # tool results are fed back with the matching call id
    last_messages = fn.calls[-1]["messages"]
    tool_msgs = [m for m in last_messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["1", "2"]
    assert fn.calls[0]["tools"] and fn.calls[0]["temperature"] == 0.0


def test_recovers_from_tool_error(db_path, scripted):
    fn = scripted(
        [
            response(tool_calls=[tool_call("1", "run_sql", query="SELECT * FROM Albums")]),
            response(tool_calls=[tool_call("2", "submit_answer", sql="SELECT COUNT(*) FROM Album", answer="3")]),
        ]
    )
    traj = make_agent(db_path, fn).run("How many albums?")
    assert traj.stop_reason == "submitted"
    assert traj.tool_errors == 1
    assert "no such table" in fn.calls[1]["messages"][-1]["content"]


def test_parallel_tool_calls_in_one_turn(db_path, scripted):
    fn = scripted(
        [
            response(tool_calls=[tool_call("a", "describe_table", table="Album"), tool_call("b", "list_tables")]),
            response(tool_calls=[tool_call("c", "submit_answer", sql="SELECT 1", answer="x")]),
        ]
    )
    traj = make_agent(db_path, fn).run("q")
    assert traj.llm_calls == 2
    assert len(traj.steps) == 3
    assert traj.prompt_tokens == 200  # usage counted once per model turn


def test_nudges_once_then_gives_up(db_path, scripted):
    fn = scripted([response(content="I think it's 3."), response(content="Still 3.")])
    traj = make_agent(db_path, fn).run("q")
    assert traj.stop_reason == "no_submission"
    assert fn.calls[1]["messages"][-1]["role"] == "user"


def test_nudge_then_submit(db_path, scripted):
    fn = scripted(
        [
            response(content="Let me think."),
