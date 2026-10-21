"""LLM-simulated caller: plays the script in text, revealing details only when asked."""

from __future__ import annotations

import json

from fastlane.eval.scenarios import CallScript
from fastlane.models.base import LLMClient
from fastlane.models.parsing import _first_json_object

CALLER_SYSTEM = """You are role-playing a CUSTOMER phoning FastLane, a booking line in Hyderabad. You are NOT the agent.
You are {name}. Speak Indian English; your manner: {style}. Keep every reply short (one or two spoken sentences).

Your goal: {goal}

Details you know - give each one only when the agent asks for it (or when it naturally fits); never list them all at once:
{facts}
Your phone number is {phone}; you are calling from it.

Rules:
- Never invent details that are not listed. If asked something you don't know, say you're not sure.
- When the agent reads back booking details, check them against your goal and details. If everything is right, say yes clearly. If anything is wrong, say exactly what is wrong.
- Stay on your goal. Accept only what your goal allows.
- When your goal is done, or the agent says it is transferring you to staff, say a short goodbye and set "done": true.
- If the agent clearly cannot help after a few attempts, say goodbye and set "done": true.

Reply with JSON only: {{"say": "<your words>", "done": true|false, "moments": [<tags>]}}
moments tags describe your reply: relative_date, festival_date, change_mind, policy_question, red_flag, rule_break, returning_caller, confirm, correct, give_detail, goodbye."""


class CallerSim:
    def __init__(self, client: LLMClient, script: CallScript, seed: int = 0, temperature: float = 0.7):
        self.client = client
        self.script = script
        self.seed = seed
        self.temperature = temperature
        facts = "\n".join(f"- {k}: {v}" for k, v in script.facts.items()) or "- (none)"
        self.system = CALLER_SYSTEM.format(name=script.caller_name, style=script.style, goal=script.goal, facts=facts, phone=script.phone)
        self.history: list[dict] = []  # from the caller's point of view: agent = user, caller = assistant
        self.latency_ms = 0.0

    def opening(self, greeting: str) -> tuple[str, list[str]]:
        self.history.append({"role": "user", "content": greeting})
        self.history.append({"role": "assistant", "content": json.dumps({"say": self.script.opening, "done": False})})
        return self.script.opening, []

    def reply(self, agent_text: str) -> tuple[str, bool, list[str]]:
        self.history.append({"role": "user", "content": agent_text})
        msgs = [{"role": "system", "content": self.system}, *self.history]
        r = self.client.chat(msgs, temperature=self.temperature, max_tokens=150, json_mode=True, seed=self.seed)
        self.latency_ms += r.latency_ms
        obj = _first_json_object(r.text or "") or {}
        say = str(obj.get("say") or "").strip() or (r.text or "").strip()[:200] or "Sorry, can you repeat that?"
        done = bool(obj.get("done"))
        moments = [m for m in obj.get("moments", []) if isinstance(m, str)] if isinstance(obj.get("moments"), list) else []
        self.history.append({"role": "assistant", "content": json.dumps({"say": say, "done": done})})
        return say, done, moments
