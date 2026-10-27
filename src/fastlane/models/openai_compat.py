"""One client for every OpenAI-compatible endpoint: Groq, Gemini, Ollama (and paid providers later).

Rate limits: free tiers cap requests and tokens per minute. The client waits when the provider says
so (429 + retry-after, or remaining-tokens headers) and records the wait as `throttle_ms`, separate
from `latency_ms`, so throttling never inflates the measured model time.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

from fastlane.models.base import LLMResponse, ToolCall, Usage
from fastlane.models.env import get_key


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str | None  # None = no key (local Ollama)
    supports_reasoning_effort: bool = False
    supports_logprobs: bool = False


PROVIDERS = {
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", supports_reasoning_effort=True),
    "gemini": Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY", supports_reasoning_effort=True),
    "ollama": Provider("ollama", "http://localhost:11434/v1", None, supports_logprobs=True),
    # `python -m mlx_lm server --model ... --adapter-path ... --port 8080` on Apple Silicon.
    "mlx": Provider("mlx", "http://localhost:8080/v1", None, supports_logprobs=True),
}

_DURATION = re.compile(r"(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")


def _seconds(value: str | None) -> float:
    """Parse '1m26.4s', '3.3s', '120ms', '7' into seconds."""
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        pass
    m = _DURATION.match(value.strip())
    if not m:
        return 0.0
    mins, secs, ms = (float(x) if x else 0.0 for x in m.groups())
    return mins * 60 + secs + ms / 1000


class RateLimited(Exception):
    """Raised instead of sleeping when the wait would exceed the client's max_wait_s (lets a fallback take over)."""

    def __init__(self, wait_s: float, message: str = ""):
        super().__init__(f"rate limited for {wait_s:.0f}s: {message}")
        self.wait_s = wait_s


class OpenAICompatClient:
    def __init__(self, provider: str, model: str, timeout: float = 60.0, max_retries: int = 6, max_wait_s: float | None = None):
        from openai import OpenAI

        self.provider = PROVIDERS[provider]
        self.model = model
        self.name = f"{provider}:{model}"
        api_key = get_key(self.provider.key_env) if self.provider.key_env else "ollama"
        self.client = OpenAI(base_url=self.provider.base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.max_retries = max_retries
        self.max_wait_s = max_wait_s
        self._lock = threading.Lock()
        self._not_before = 0.0  # monotonic time before which we should not send (token budget exhausted)

    def _respect_budget(self) -> float:
        with self._lock:
            wait = self._not_before - time.monotonic()
        if self.max_wait_s is not None and wait > self.max_wait_s:
            raise RateLimited(wait, "token budget")
        if wait > 0:
            time.sleep(wait)
            return wait * 1000
        return 0.0

    def _note_headers(self, headers) -> None:
        remaining = headers.get("x-ratelimit-remaining-tokens")
        if remaining is not None and remaining.isdigit() and int(remaining) < 2500:
            reset = _seconds(headers.get("x-ratelimit-reset-tokens"))
            with self._lock:
                self._not_before = max(self._not_before, time.monotonic() + reset)

    def _adapt(self, messages: list[dict]) -> list[dict]:
        """Make a history built with another provider acceptable to this one.

        Gemini needs its own thought_signature on every tool call it is shown, so tool exchanges made by
        another model are rewritten as plain text. Other providers get provider-specific fields removed."""
        if not any(m.get("tool_calls") for m in messages):
            return messages
        out: list[dict] = []
        flattened: set[str] = set()
        for m in messages:
            calls = m.get("tool_calls")
            if m.get("role") == "assistant" and calls:
                if self.provider.name == "gemini" and any("extra_content" not in c for c in calls):
                    flattened |= {c["id"] for c in calls}
                    text = "; ".join(f"called {c['function']['name']}({c['function']['arguments']})" for c in calls)
                    out.append({"role": "assistant", "content": ((m.get("content") or "") + f" [{text}]").strip()})
                    continue
                if self.provider.name != "gemini":
                    m = {**m, "tool_calls": [{k: c[k] for k in ("id", "type", "function") if k in c} for c in calls]}
            elif m.get("role") == "tool" and m.get("tool_call_id") in flattened:
                out.append({"role": "user", "content": f"(tool result) {m['content']}"})
                continue
            out.append(m)
        return out

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=512, json_mode=False, reasoning=None, logprobs=False, seed=None) -> LLMResponse:
        from openai import APIConnectionError, APIStatusError, RateLimitError

        kwargs: dict = {"model": self.model, "messages": self._adapt(messages), "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if reasoning and self.provider.supports_reasoning_effort:
            kwargs["reasoning_effort"] = reasoning
        if seed is not None:
            kwargs["seed"] = seed
        if logprobs and self.provider.supports_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5

        throttle_ms = 0.0
        for attempt in range(self.max_retries + 1):
            throttle_ms += self._respect_budget()
            t0 = time.perf_counter()
            try:
                raw = self.client.chat.completions.with_raw_response.create(**kwargs)
            except RateLimitError as e:
                wait = _seconds(e.response.headers.get("retry-after")) or min(60.0, 2.0 ** attempt)
                self.last_rate_limit = str(e)[:300]
                if self.max_wait_s is not None and wait > self.max_wait_s:
                    with self._lock:
                        self._not_before = max(self._not_before, time.monotonic() + wait)
                    raise RateLimited(wait, self.last_rate_limit) from None
                if wait > 60:
                    import sys

                    print(f"[{self.name}] rate limited, waiting {wait:.0f}s: {self.last_rate_limit}", file=sys.stderr, flush=True)
                time.sleep(wait)
                throttle_ms += wait * 1000
                continue
            except (APIConnectionError, APIStatusError) as e:
                status = getattr(e, "status_code", None)
                if attempt < self.max_retries and (status is None or status >= 500):
                    time.sleep(min(10.0, 1.5 ** attempt))
                    continue
                raise
            latency_ms = (time.perf_counter() - t0) * 1000
            self._note_headers(raw.headers)
            return self._convert(raw.parse(), latency_ms, throttle_ms)
        raise RuntimeError(f"{self.name}: still rate limited after {self.max_retries} retries")

    def _convert(self, resp, latency_ms: float, throttle_ms: float) -> LLMResponse:
        choice = resp.choices[0]
        msg = choice.message
        calls = [ToolCall.parse(tc.id, tc.function.name, tc.function.arguments,
                                tc.model_dump(exclude={"id", "type", "function", "index"}, exclude_none=True))
                 for tc in (msg.tool_calls or [])]
        u = resp.usage
        cached = 0
        if u is not None and getattr(u, "prompt_tokens_details", None) is not None:
            cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
        usage = Usage(u.prompt_tokens if u else 0, u.completion_tokens if u else 0, cached)
        top = None
        lp = getattr(choice, "logprobs", None)
        if lp is not None and getattr(lp, "content", None):
            top = [{alt.token: alt.logprob for alt in (tok.top_logprobs or [])} or {tok.token: tok.logprob} for tok in lp.content[:12]]
        return LLMResponse(
            text=msg.content, tool_calls=calls, usage=usage, latency_ms=latency_ms, throttle_ms=throttle_ms,
            model=self.name, reasoning=getattr(msg, "reasoning", None), top_logprobs=top, finish_reason=choice.finish_reason,
        )


class FallbackClient:
    """Primary model, with a backup for calls the primary cannot serve soon (rate limits).

    While the primary is blocked, every call goes to the backup; the primary is tried again once its
    limit has reset. `LLMResponse.model` says which model answered, so results can be split by model.
    """

    def __init__(self, primary: OpenAICompatClient, backup: OpenAICompatClient, switch_after_s: float = 10.0):
        primary.max_wait_s = switch_after_s
        self.primary, self.backup = primary, backup
        self.name = f"{primary.name}|{backup.name}"
        self._blocked_until = 0.0
        self.backup_calls = 0
        self.primary_calls = 0

    def chat(self, messages, **kwargs) -> LLMResponse:
        if time.monotonic() >= self._blocked_until:
            try:
                r = self.primary.chat(messages, **kwargs)
                self.primary_calls += 1
                return r
            except RateLimited as e:
                self._blocked_until = time.monotonic() + e.wait_s
                import sys

                print(f"[{self.name}] {self.primary.name} limited for {e.wait_s:.0f}s; using {self.backup.name}", file=sys.stderr, flush=True)
        self.backup_calls += 1
        t0 = time.perf_counter()
        try:
            return self.backup.chat(messages, **kwargs)
        except Exception as e:  # backup too slow or failing (pilot v3: one Gemini reply took 45 s): wait for the primary
            import sys

            lost_ms = (time.perf_counter() - t0) * 1000
            print(f"[{self.name}] backup failed ({type(e).__name__}) after {lost_ms / 1000:.0f}s; waiting for {self.primary.name}",
                  file=sys.stderr, flush=True)
            saved, self.primary.max_wait_s = self.primary.max_wait_s, None
            try:
                r = self.primary.chat(messages, **kwargs)
            finally:
                self.primary.max_wait_s = saved
            r.throttle_ms += lost_ms  # time lost on the backup counts as waiting, not as model latency
            return r
