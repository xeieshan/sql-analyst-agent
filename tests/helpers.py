"""Builders for fake LiteLLM responses."""

import json
from types import SimpleNamespace


def tool_call(id_: str, name: str, **args):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def response(content=None, tool_calls=None, prompt_tokens=100, completion_tokens=20):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

# Maintenance pass: 2026-04-24 09:03
