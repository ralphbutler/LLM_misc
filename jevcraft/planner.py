"""Opus: the slow, deliberate half. System 2 to Jev's System 1.

Two jobs, both rare:
  plan()    once per episode - a short ordered plan for the goal, which Jev is
            shown on every call
  decide()  only when Jev's top choice comes back under the confidence
            threshold - Opus picks from the SAME menu Jev saw

Structured output constrains `choice` to an enum of the menu's option keys, so
Opus cannot answer with something that is not on the menu. Refusals fall back
server-side ("default" mode); a refusal that survives the fallback chain returns
None and the caller keeps Jev's pick.
"""

from __future__ import annotations

import json
import random
import time

MODEL = "claude-opus-5"
PRICE_IN, PRICE_OUT = 5.00 / 1e6, 25.00 / 1e6          # $/token, claude-opus-5
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = (
    "You advise a player in Crafter, a 2D survival game seen from above. You are "
    "consulted only when a fast model was unsure. You get the situation as JSON "
    "and a menu of options that are all currently possible. Pick the single option "
    "that best advances the goal while keeping the player alive, and give a "
    "one-sentence reason a viewer can read in two seconds."
)


class OpusError(RuntimeError):
    pass


class Opus:
    def __init__(self, effort="medium", max_calls=20):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.effort = effort
        self.max_calls = max_calls
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.seconds = 0.0

    def _ask(self, user_text, schema):
        if self.calls >= self.max_calls:
            raise OpusError("Opus call budget exhausted (%d)" % self.max_calls)
        t0 = time.monotonic()
        try:
            resp = self.client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM,
                messages=[{"role": "user", "content": user_text}],
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": schema}},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except self.anthropic.APIStatusError as e:
            raise OpusError("Opus API error %s: %s" % (e.status_code, e.message))
        except self.anthropic.APIConnectionError:
            raise OpusError("Opus unreachable (network)")
        dt = time.monotonic() - t0
        self.calls += 1
        self.seconds += dt
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            return None, dt
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            return None, dt
        return json.loads(text), dt

    def decide(self, situation, options):
        """options: list of (key, label). Returns (key, reason, seconds) or None."""
        keys = [k for k, _ in options]
        schema = {
            "type": "object",
            "properties": {"choice": {"type": "string", "enum": keys},
                           "reason": {"type": "string"}},
            "required": ["choice", "reason"],
            "additionalProperties": False,
        }
        text = json.dumps({"situation": situation,
                           "options": {k: lab for k, lab in options}}, indent=1)
        data, dt = self._ask(text, schema)
        if not data or data.get("choice") not in keys:
            return None
        return data["choice"], data["reason"].strip(), dt

    def plan(self, goal_text, situation):
        schema = {
            "type": "object",
            "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
            "required": ["steps"],
            "additionalProperties": False,
        }
        text = json.dumps({"task": "Write a short ordered plan (3 to 6 steps, a few words "
                                   "each) for this goal: %s. Only steps the player can act on." % goal_text,
                           "situation": situation}, indent=1)
        data, _ = self._ask(text, schema)
        return (data or {}).get("steps", [])[:6]

    def stats(self):
        return {"model": MODEL, "effort": self.effort, "calls": self.calls,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "seconds": round(self.seconds, 1),
                "cost_usd": round(self.input_tokens * PRICE_IN + self.output_tokens * PRICE_OUT, 4)}


class FakeOpus:
    """--dry: same interface, no network, no cost."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.calls = 0

    def decide(self, situation, options):
        self.calls += 1
        k = self.rng.choice(options)[0]
        return k, "fake Opus (dry run) picked this at random", 0.0

    def plan(self, goal_text, situation):
        self.calls += 1
        return ["collect wood", "place a table", "craft a wood pickaxe", "mine stone",
                "craft the stone pickaxe"]

    def stats(self):
        return {"model": "fake", "calls": self.calls, "cost_usd": 0.0, "seconds": 0.0}
