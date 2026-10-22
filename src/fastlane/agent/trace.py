"""What happened in each turn: who answered, why, how long the models took, what it cost."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from fastlane.models.base import LLMResponse


@dataclass
class ModelCall:
    role: str  # small | large | router
    model: str
    purpose: str  # step | route | reply
    latency_ms: float
    throttle_ms: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0

    @classmethod
    def of(cls, role: str, purpose: str, r: LLMResponse) -> "ModelCall":
        return cls(role, r.model, purpose, round(r.latency_ms, 1), round(r.throttle_ms, 1),
                   r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.cached_tokens)


@dataclass
class ToolRecord:
    tool: str
    args: dict
    ok: bool
    error: str | None
    ms: float
    by: str  # small | large | orchestrator


@dataclass
class TurnTrace:
    turn: int
    caller: str
    agent: str = ""
    handled_by: str = "small"  # small | large | staff | template
    escalation_reason: str | None = None
    escalation_score: float | None = None
    small_route: str | None = None  # what the small model itself decided
    model_calls: list[ModelCall] = field(default_factory=list)
    tools: list[ToolRecord] = field(default_factory=list)
    wall_ms: float = 0.0
    policy_prefetched: bool = False
    ended: bool = False

    @property
    def model_ms(self) -> float:
        return round(sum(c.latency_ms for c in self.model_calls), 1)

    @property
    def large_calls(self) -> int:
        return sum(1 for c in self.model_calls if c.role in ("large", "router"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["model_ms"] = self.model_ms
        return d
