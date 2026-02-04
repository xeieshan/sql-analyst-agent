"""Builders for fake LiteLLM responses."""

import json
from types import SimpleNamespace


def tool_call(id_: str, name: str, **args):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def response(content=None, tool_calls=None, prompt_tokens=100, completion_tokens=20):
