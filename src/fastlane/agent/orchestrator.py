"""The per-turn loop: who answers each caller turn, and how.

FastLane (router="small"): the small model handles the turn step by step (JSON out, one tool per
step). The turn moves to the large model when
  * the small model routes it to "llm" or its escalation score exceeds the threshold,
  * a tool returns an error or no availability,
  * the caller changed an earlier choice (a booking-record field was overwritten),
  * a policy search is not confident,
and to staff when the small model routes to "staff" or two tool calls fail in a row.
The large model gets the booking record, the last few turns and the same tools, finishes the
turn, and control returns to the small model.

Baselines: B1 always_large, B2 always_small (no large model), B3 large_prompt (a large-model
router call before each turn). Every other difference between systems is a SystemConfig flag.
"""

from __future__ import annotations

import re
import time
from datetime import date
from difflib import SequenceMatcher
from typing import Any

from fastlane.agent import templates
from fastlane.agent.trace import ModelCall, ToolRecord, TurnTrace
from fastlane.config import ESCALATION_THRESHOLD, FASTLANE, SystemConfig
from fastlane.gateway.dates import parse_time, resolve_date
from fastlane.gateway.gateway import Gateway
from fastlane.gateway.record import Passenger
from fastlane.gateway.tools import WRITE_TOOLS
from fastlane.models import prompts
from fastlane.models.base import LLMClient, openai_tools
from fastlane.models.parsing import escalation_score, parse_small
from fastlane.retrieval.early_lookup import EarlyLookup, PrefetchingSearcher
from fastlane.retrieval.index import PolicyIndex
from fastlane.world.db import World

ESCALATING_ERRORS = {
    "system_error", "no_availability", "not_found", "ambiguous", "not_released", "rule_violation", "not_understood",
    "invalid_args", "identity", "not_found_or_mismatch", "not_active", "no_change", "unknown_tool", "unavailable",
}
STAFF_REASONS = {
    "payment": ("payment", "refund", "charge", "money", "upi", "card", "deposit waiver", "waive"),
    "complaint": ("complain", "complaint", "rude", "dirty", "bad", "angry"),
    "large_group": ("group", "limit", "too many", "large party"),
    "identity": ("identity", "phone", "someone else", "not theirs", "verify"),
    "caller_request": ("human", "person", "staff", "manager", "agent"),
    "out_of_scope": ("out of scope", "flight", "train", "hotel", "not offered", "outside"),
}
POLICY_CONFIDENCE_MIN = 0.35


class Agent:
    def __init__(
        self,
        world: World,
        caller_phone: str,
        config: SystemConfig = FASTLANE,
        small: LLMClient | None = None,
        large: LLMClient | None = None,
        policy_index: PolicyIndex | None = None,
        inject_failures: dict[str, int] | None = None,
        threshold: float = ESCALATION_THRESHOLD,
        max_small_steps: int = 4,
        max_large_steps: int = 8,
        small_context_lines: int = 6,
        large_context_lines: int = 8,
        raise_errors: bool = False,
        compact_small_prompt: bool = False,
    ):
        self.cfg = config
        self.small = small
        self.large = large if config.large_model else None
        if config.router in ("small", "large_prompt", "always_small") and small is None:
            raise ValueError(f"{config.name} needs a small model")
        if config.router in ("always_large", "large_prompt") and self.large is None:
            raise ValueError(f"{config.name} needs a large model")
        index = policy_index or PolicyIndex()
        self.early = EarlyLookup(index, enabled=config.early_lookup)
        self.gw = Gateway(world, caller_phone, config, policy=PrefetchingSearcher(index, self.early), inject_failures=inject_failures)
        self.threshold = threshold
        self.max_small_steps = max_small_steps
        self.max_large_steps = max_large_steps
        self.small_context_lines = small_context_lines
        self.large_context_lines = large_context_lines
        self.turns: list[tuple[str, str]] = []
        self.traces: list[TurnTrace] = []
        self.history: dict | None = None
        self.pending_call: tuple[str, dict] | None = None
        self.ended = False
        self.raise_errors = raise_errors
        self._specs = self.gw.specs()
        # The fine-tuned small model uses the short prompt it was trained with; prompted models get the full one.
        self._small_system = prompts.small_system(config, self._specs, compact=compact_small_prompt)
        self._large_system = prompts.large_system(config)
        self._large_tools = openai_tools(self._specs)

    # ================================================================ call flow
    def greet(self) -> str:
        tr = TurnTrace(turn=0, caller="", handled_by="template")
        if self.cfg.caller_history:
            # Caller-ID lookup starts before the caller says anything (early lookup of history).
            res = self._tool("get_caller_history", {}, "orchestrator", tr)
            self.history = res if res.get("ok") else None
        tr.agent = templates.GREETING
        self.traces.append(tr)
        self.turns.append(("Agent", templates.GREETING))
        self.gw.end_turn(templates.GREETING)
        return templates.GREETING

    def respond(self, caller_text: str, partials: list[str] | None = None) -> str:
        t0 = time.perf_counter()
        tr = TurnTrace(turn=self.gw.turn + 1, caller=caller_text)
        for p in partials or []:
            self.early.on_partial(p)
        tr.policy_prefetched = self.early.on_final(caller_text) is not None
        self.gw.begin_turn(caller_text)
        self.turns.append(("Caller", caller_text))
        try:
            text = self._handle(tr, caller_text)
        except Exception as e:  # a crashed turn must not crash the experiment; log it and recover
            if self.raise_errors:
                raise
            tr.escalation_reason = f"agent error: {type(e).__name__}: {e}"[:300]
            text = templates.FALLBACK
        if templates.is_goodbye(caller_text) and self.gw.pending is None:
            self.ended = True
        tr.ended = self.ended
        tr.agent = text
        tr.wall_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.traces.append(tr)
        self.turns.append(("Agent", text))
        self.gw.end_turn(text)
        return text

    def close(self) -> None:
        self.early.close()

    # ================================================================ routing
    def _handle(self, tr: TurnTrace, caller_text: str) -> str:
        cfg = self.cfg
        # 1. The caller just confirmed a read-back: execute exactly the pending action, no model needed.
        if cfg.templates and self.gw.confirmed_digest and self.pending_call:
            name, args = self.pending_call
            res = self._tool(name, args, "orchestrator", tr)
            if res.get("ok"):
                self.pending_call = None
                tr.handled_by = "template"
                return templates.done(name, res["booking"], res)
        if self.gw.record.readback_status == "rejected":
            self.pending_call = None
        # 2. Goodbye with nothing pending.
        if cfg.templates and templates.is_goodbye(caller_text) and self.gw.pending is None:
            tr.handled_by = "template"
            return templates.GOODBYE
        # 3. The booking system keeps failing: staff.
        if self.gw.consecutive_failures >= 2:
            return self._staff(tr, "repeated_failure")
        # 4. Route the turn.
        if cfg.router == "always_large":
            return self._large(tr, None, [])
        if cfg.router == "always_small":
            return self._small(tr, use_model_route=False)
        if cfg.router == "large_prompt":
            if self._router_says_hard(tr):
                return self._large(tr, "large-model router: hard", [])
            return self._small(tr, use_model_route=False)
        return self._small(tr, use_model_route=True)

    def _router_says_hard(self, tr: TurnTrace) -> bool:
        msgs = [{"role": "system", "content": prompts.ROUTER_SYSTEM},
                {"role": "user", "content": prompts.transcript(self.turns, 6) + "\n\nIs the caller's latest turn EASY or HARD?"}]
        r = self.large.chat(msgs, max_tokens=64, reasoning="low")
        tr.model_calls.append(ModelCall.of("router", "route", r))
        return "EASY" not in (r.text or "").upper()

    # ================================================================ small model
    def _small_messages(self, steps: list[tuple[str, str, dict]]) -> list[dict]:
        cfg = self.cfg
        record = self.gw.record.render() if cfg.booking_record else None
        policy = self.early.ready.result if self.early.ready is not None else None
        ctx = prompts.context_block(record, self.history if cfg.caller_history else None, policy)
        limit = self.small_context_lines if cfg.booking_record else None  # A4: full chat history instead of the record
        user = (ctx + "\n\n" if ctx else "") + "CONVERSATION:\n" + prompts.transcript(self.turns, limit) + "\n\nOutput the JSON for the agent's next step."
        msgs = [{"role": "system", "content": self._small_system}, {"role": "user", "content": user}]
        for out, name, res in steps:
            msgs.append({"role": "assistant", "content": out})
            msgs.append({"role": "user", "content": f"TOOL RESULT {name}: {prompts.compact_result(res)}"})
        return msgs

    def _small(self, tr: TurnTrace, use_model_route: bool) -> str:
        large_ok = self.large is not None
        steps: list[tuple[str, str, dict]] = []
        for step in range(self.max_small_steps):
            r = self.small.chat(self._small_messages(steps), max_tokens=350, json_mode=True, logprobs=use_model_route)
            tr.model_calls.append(ModelCall.of("small", "step", r))
            p = parse_small(r.text)
            score = escalation_score(r, p)
            if step == 0:
                tr.small_route, tr.escalation_score = p.route, score
            changed = self._apply_slots(p.slots)
            if p.route == "staff":
                return self._staff(tr, p.reason or "other")
            reason = None
            if use_model_route and (p.route == "llm" or score > self.threshold or not p.ok):
                reason = "unparseable small output" if not p.ok else f"small model: {p.reason or 'hard turn'}"
            if changed:
                reason = reason or f"caller changed {', '.join(changed)}"
            if reason and large_ok and (use_model_route or changed):
                return self._large(tr, reason, steps)
            if p.tool:
                name, args = p.tool["name"], p.tool["args"]
                res = self._tool(name, args, "small", tr)
                steps.append((r.text or "", name, res))
                outcome = self._after_tool(tr, name, args, res)
                if outcome is not None:
                    return outcome
                trigger = self._tool_trigger(name, res)
                if trigger and large_ok:
                    return self._large(tr, trigger, steps)
                continue
            if p.say:
                if large_ok and use_model_route and self._numeric_policy_answer(p.say, steps):
                    return self._large(tr, "policy answer with numbers", steps)
                return p.say
            if large_ok and use_model_route:
                return self._large(tr, "small model gave no reply", steps)
            return templates.FALLBACK
        if large_ok:
            return self._large(tr, "too many small-model steps", steps)
        return templates.FALLBACK

    def _after_tool(self, tr: TurnTrace, name: str, args: dict, res: dict) -> str | None:
        """Shared handling of tool results that end the turn (hand-off, read-back, completed write)."""
        if name == "handoff_to_staff" and res.get("ok"):
            tr.handled_by = "staff"
            self.ended = True
            return templates.HANDOFF
        if self.gw.consecutive_failures >= 2:
            return self._staff(tr, "repeated_failure")
        if res.get("error") == "needs_readback":
            self.pending_call = (name, args)
            if self.cfg.templates:
                return res["readback"]
        if res.get("ok") and name in WRITE_TOOLS:
            self.pending_call = None
            if self.cfg.templates:
                return templates.done(name, res["booking"], res)
        return None

    @staticmethod
    def _numeric_policy_answer(say: str, steps: list[tuple[str, str, dict]]) -> bool:
        """A policy answer with amounts, percentages or hours is worked out by the large model from the same search
        result (pilot v3: the small model read a refund table wrongly, 90% instead of 25%)."""
        return any(name == "search_policy" for _, name, _ in steps) and bool(re.search(r"\d|%|\bpercent", say))

    @staticmethod
    def _tool_trigger(name: str, res: dict) -> str | None:
        err = res.get("error")
        if err == "no_availability":
            return "no availability"
        if err in ESCALATING_ERRORS:
            return f"tool error: {err}"
        if res.get("ok") and name in ("search_inventory", "find_alternatives"):
            if res.get("exact_match_available") is False or res.get("found") is False:
                return "no availability"
        if res.get("ok") and name == "search_policy":
            if not res.get("covered") or res.get("confidence", 1.0) < POLICY_CONFIDENCE_MIN:
                return "unconfident policy match"
        return None

    # ---------------------------------------------------------------- slots
    def _canonical(self, key: str, value: Any) -> Any:
        """Map what the small model heard onto canonical values, so spelling variants are not 'changes'."""
        cat = self.gw.inv.cat
        if not isinstance(value, (str, int)) or value in ("", None):
            return None
        if key == "date":
            try:
                return date.fromisoformat(str(value)).isoformat()
            except ValueError:
                pass
            if not self.cfg.date_tools:
                return None  # without date tools a spoken date is the large model's job (variant A2)
            r = resolve_date(str(value), self.gw.now.date())  # the caller's words, resolved in code
            return r.date.isoformat() if r.date and r.in_window else None
        if key == "time":
            window = parse_time(str(value))
            return f"{window[0] // 60:02d}:{window[0] % 60:02d}" if window and window[0] == window[1] else None
        if key in ("quantity", "party_size"):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        if key == "domain":
            v = str(value).lower()
            return next((d for d, words in (("movies", ("movie", "film", "cinema", "show", "ticket")),
                                            ("buses", ("bus", "travel", "trip")),
                                            ("restaurants", ("restaurant", "table", "dinner", "lunch", "food")))
                         if any(w in v for w in words)), None)
        if key == "intent":
            v = str(value).lower()
            return next((i for i, words in (("cancel", ("cancel",)), ("modify", ("change", "modify", "reschedule", "move")),
                                            ("book", ("book", "reserve", "buy")), ("info", ("info", "question", "ask", "policy")))
                         if any(w in v for w in words)), None)
        if key == "berth_preference":
            v = str(value).lower()
            return next((b for b in ("lower", "upper", "window") if b in v), None)
        finder = {"movie": cat.movie, "cinema": cat.cinema, "restaurant": cat.restaurant, "seat_category": cat.seat_category,
                  "origin": cat.city, "destination": cat.city, "bus_type": cat.bus_type}.get(key)
        if finder is not None:
            match, _ = finder(str(value))
            if match is None:
                return None
            return match.id if key in ("seat_category", "origin", "destination", "bus_type") else match.label
        return value

    def _apply_slots(self, slots: dict) -> list[str]:
        rec = self.gw.record
        updates: dict[str, Any] = {}
        for key, value in slots.items():
            if key == "passengers" and isinstance(value, list):
                pax = []
                for p in value:
                    if isinstance(p, dict) and p.get("name"):
                        try:
                            pax.append(Passenger(**{k: p.get(k) for k in ("name", "age", "gender")}))
                        except Exception:
                            pax.append(Passenger(name=str(p["name"])))
                if pax:
                    rec.set(track=False, passengers=pax)
                continue
            if key not in type(rec).model_fields or key in ("passengers", "readback_status", "completed_actions", "changed_fields"):
                continue
            canon = self._canonical(key, value)
            if canon is None:
                continue
            old = getattr(rec, key)
            if isinstance(old, str) and isinstance(canon, str) and SequenceMatcher(None, old.lower(), canon.lower()).ratio() > 0.85:
                continue
            updates[key] = canon
        if not updates:
            return []
        try:
            return rec.set(**updates)
        except Exception:
            return []

    # ================================================================ large model
    def _large_messages(self, steps: list[tuple[str, str, dict]]) -> list[dict]:
        cfg = self.cfg
        full_history = cfg.router == "always_large" or not cfg.booking_record
        record = self.gw.record.render() if cfg.booking_record else None
        policy = self.early.ready.result if self.early.ready is not None else None
        ctx = prompts.context_block(record, self.history if cfg.caller_history else None, policy)
        prior = self.turns[:-1] if full_history else self.turns[:-1][-self.large_context_lines:]
        msgs: list[dict] = [{"role": "system", "content": self._large_system}]
        for who, text in prior:
            msgs.append({"role": "user" if who == "Caller" else "assistant", "content": text})
        latest = self.turns[-1][1]
        content = (ctx + "\n\n" if ctx else "") + f"Caller: {latest}"
        if steps:
            work = "\n".join(f"- {name} -> {prompts.compact_result(res, 900)}" for _, name, res in steps)
            content += f"\n\nWork already done this turn by the assistant (tool results):\n{work}"
        msgs.append({"role": "user", "content": content})
        return msgs

    def _large(self, tr: TurnTrace, reason: str | None, steps: list[tuple[str, str, dict]]) -> str:
        tr.handled_by = "large"
        tr.escalation_reason = reason
        msgs = self._large_messages(steps)
        effort = "medium" if self.cfg.large_reasoning else "low"
        for _ in range(self.max_large_steps):
            try:
                r = self.large.chat(msgs, tools=self._large_tools, max_tokens=1200, reasoning=effort)
            except Exception as e:
                # Providers validate tool arguments server-side (HTTP 400). Tell the model and let it retry.
                if "tool call validation" not in str(e).lower() and "failed to call a function" not in str(e).lower():
                    raise
                msgs.append({"role": "user", "content": f"(System) Your last tool call was rejected: {str(e)[:300]}. Fix the arguments and try again."})
                continue
            tr.model_calls.append(ModelCall.of("large", "reply", r))
            if not r.tool_calls:
                return (r.text or "").strip() or templates.FALLBACK
            msgs.append({"role": "assistant", "content": r.text or "", "tool_calls": [tc.as_message_part() for tc in r.tool_calls]})
            outcome = None
            for tc in r.tool_calls:
                if outcome is not None:  # turn already decided; answer remaining calls so the transcript stays valid
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": '{"ok":false,"error":"skipped"}'})
                    continue
                res = self._tool(tc.name, tc.args, "large", tr)
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": prompts.compact_result(res)})
                outcome = self._after_tool(tr, tc.name, tc.args, res)
            if outcome is not None:
                return outcome
        return templates.FALLBACK

    # ================================================================ staff and tools
    def _staff(self, tr: TurnTrace, reason: str) -> str:
        code = "other"
        low = reason.lower()
        if low in {"payment", "complaint", "out_of_scope", "large_group", "repeated_failure", "identity", "caller_request"}:
            code = low
        else:
            for key, words in STAFF_REASONS.items():
                if any(w in low for w in words):
                    code = key
                    break
        caller = self.turns[-1][1] if self.turns else ""
        summary = templates.handoff_summary(self.gw.record.render(), caller, reason)
        self._tool("handoff_to_staff", {"reason": code, "summary": summary}, "orchestrator", tr)
        tr.handled_by = "staff"
        tr.escalation_reason = tr.escalation_reason or f"staff: {reason}"
        self.ended = True
        return templates.HANDOFF

    def _record_from_args(self, name: str, args: dict) -> None:
        """Details the models pass to write tools (name, passengers, sizes) go into the record, so they
        survive a switch between models even when the call itself was refused."""
        if name not in WRITE_TOOLS or not isinstance(args, dict):
            return
        slots = {k: args[k] for k in ("passengers", "quantity", "party_size", "special_request", "booking_id") if args.get(k)}
        if args.get("name"):
            slots["caller_name"] = args["name"]
        rec = self.gw.record
        if isinstance(slots.get("passengers"), list):
            self._apply_slots({"passengers": slots.pop("passengers")})
        for key in ("caller_name", "special_request", "booking_id"):
            if isinstance(slots.get(key), str):
                rec.set(track=False, **{key: slots.pop(key)})
        for key in ("quantity", "party_size"):
            if key in slots:
                try:
                    rec.set(track=False, **{key: int(slots[key])})
                except (TypeError, ValueError):
                    pass

    def _tool(self, name: str, args: dict, by: str, tr: TurnTrace) -> dict:
        self._record_from_args(name, args)
        t0 = time.perf_counter()
        res = self.gw.call(name, args)
        if res.get("error") == "system_error" and self.cfg.tool_retry:
            tr.tools.append(ToolRecord(name, args, False, "system_error", round((time.perf_counter() - t0) * 1000, 2), by))
            t0 = time.perf_counter()
            res = self.gw.call(name, args)
        tr.tools.append(ToolRecord(name, args, bool(res.get("ok")), res.get("error"), round((time.perf_counter() - t0) * 1000, 2), by))
        return res
