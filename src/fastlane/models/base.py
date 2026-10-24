"""Provider-neutral model interface.

Every model call returns an `LLMResponse` with its own latency, so per-turn model time (the main
speed metric) is the sum of `latency_ms` of the calls made in that turn. Time spent waiting on a
provider's rate limit is recorded separately in `throttle_ms` and is not model time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    raw_args: str = ""
    # Provider-specific fields that must be sent back with the call in the next request
    # (e.g. Gemini's extra_content.google.thought_signature).
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, id: str, name: str, raw: str | dict | None, extra: dict | None = None) -> "ToolCall":
        if isinstance(raw, dict):
            return cls(id, name, raw, json.dumps(raw), extra or {})
        try:
            args = json.loads(raw or "{}")
            if not isinstance(args, dict):
                args = {}
        except json.JSONDecodeError:
            args = {"_unparsed": raw}
        return cls(id, name, args, raw or "", extra or {})

    def as_message_part(self) -> dict:
        """The tool call as it must appear in the assistant message of the next request."""
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.raw_args or "{}"}, **self.extra}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + other.prompt_tokens, self.completion_tokens + other.completion_tokens,
                     self.cached_tokens + other.cached_tokens)


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    throttle_ms: float = 0.0  # waiting on rate limits; not model time
    model: str = ""
    reasoning: str | None = None
    # Log-probabilities of the first generated tokens, when the provider returns them:
    # list of {token: logprob} dicts (top alternatives at each position).
    top_logprobs: list[dict[str, float]] | None = None
    finish_reason: str | None = None


class LLMClient(Protocol):
    name: str

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
        reasoning: str | None = None,  # "low" | "medium" | "high" where supported
        logprobs: bool = False,
        seed: int | None = None,
    ) -> LLMResponse: ...


def openai_tools(specs: list[dict]) -> list[dict]:
    """Gateway specs (name, description, input_schema) -> OpenAI `tools` format."""
    return [{"type": "function", "function": {"name": s["name"], "description": s["description"], "parameters": s["input_schema"]}}
            for s in specs]
