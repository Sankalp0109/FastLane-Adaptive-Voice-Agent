"""Deterministic stand-in model for tests: replies come from a script or a function."""

from __future__ import annotations

import json
from typing import Callable

from fastlane.models.base import LLMResponse, ToolCall, Usage


class ScriptedClient:
    """`script` items are either a string (text reply), a dict {"tool": name, "args": {...}} for a
    native tool call, or a callable(messages, tools) -> LLMResponse. Each chat() consumes one item."""

    def __init__(self, script: list | Callable | None = None, name: str = "scripted", latency_ms: float = 100.0):
        self.script = script if script is not None else []
        self.name = name
        self.latency_ms = latency_ms
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=512, json_mode=False, reasoning=None, logprobs=False, seed=None) -> LLMResponse:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools, "reasoning": reasoning})
        if callable(self.script):
            item = self.script(messages, tools)
        elif self.script:
            item = self.script.pop(0)
        else:
            raise AssertionError(f"{self.name}: script exhausted")
        if isinstance(item, LLMResponse):
            return item
        usage = Usage(sum(len(str(m.get("content", ""))) // 4 for m in messages), 20)
        if isinstance(item, dict) and "tool" in item:
            call = ToolCall(f"call_{len(self.calls)}", item["tool"], item.get("args", {}), json.dumps(item.get("args", {})))
            return LLMResponse(None, [call], usage, self.latency_ms, model=self.name, finish_reason="tool_calls")
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMResponse(text, [], usage, self.latency_ms, model=self.name, finish_reason="stop")
