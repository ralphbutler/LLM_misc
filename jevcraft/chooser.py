"""The Jev agent: at each decision point, one API call picks the next option.

Everything numeric is turned into words or yes/no in code before Jev sees it
(the jevsearch finding: Jev judges structure well and arithmetic badly). Three
questions per call, answered in parallel from the same state:

  next_step       choice over the feasible options   -> the decision
  in_danger       noul                                -> red annotation in the video
  should_recover  noul                                -> annotation (drink/eat/sleep soon?)

Options are shuffled per call: upstream Probe 4b found a 44% top-choice flip
rate from option order alone.
"""

from __future__ import annotations

import random

from agents import Decision, ScriptedAgent
from skills import GOAL_TOOLS, remaining
from client import JevClient, JevError
from state import VIEW, dijkstra, neighbors

RULES = (
    "Crafter, a 2D survival game seen from above. Chop trees for wood. A crafting "
    "table costs wood and must be within one tile of you to craft. A wood pickaxe "
    "lets you mine stone; a stone pickaxe needs wood and stone at the table. "
    "Zombies walk toward you and hit for 2 health every few steps while adjacent "
    "(7 if you are asleep); without a sword a zombie takes 5 hits to kill. "
    "Nights bring many more zombies. Health slowly returns while food, drink and "
    "energy are not empty."
)

GOAL_TEXT = {"stone_pickaxe": "make a stone pickaxe", "iron_pickaxe": "make an iron pickaxe"}

RULES_IRON = (
    " Coal needs a wood pickaxe; iron needs a stone pickaxe, and both are usually "
    "inside rock, so you dig to them. The furnace costs stone. The iron pickaxe "
    "needs the table AND the furnace both within one tile. A sword makes zombies "
    "die in fewer hits. Lava kills instantly if you walk into it. Zombies cannot "
    "dig, so a room walled in with stone is safe at night."
)


def _level(v):
    return "fine" if v >= 7 else "getting low" if v >= 4 else "critical"


def _dist_words(n):
    if n is None:
        return None
    return ("right here" if n <= 1 else "close (2-5 steps)" if n <= 5
            else "a short walk" if n <= 12 else "far")


def _time_words(daylight):
    return "day" if daylight > 0.7 else "evening, getting dark" if daylight > 0.45 else "night"


def situation(st, goal, last_outcome, plan=None):
    """The state Jev is shown. No raw distances or sums - words and booleans."""
    need = remaining(st, goal)

    def how_far(material):
        if not st.known_any(material):
            return "not seen yet"
        path = dijkstra(st, lambda u: any(
            st.known.inside(n) and st.known.mat[n] == material for n in neighbors(u)))
        return _dist_words(len(path)) if path is not None else "seen, but no path to it"

    threats = []
    for d, _ in st.hostiles():
        threats.append("zombie adjacent" if d == 1 else "zombie very close (2-3 steps)"
                       if d <= 3 else "zombie nearby")
    inv = {k: v for k, v in st.inv.items()
           if v and k not in ("health", "food", "drink", "energy")}
    def enough(item):
        have = st.inv.get(item, 0)
        return "enough for the goal" if have >= need[item] else "%d more needed" % (need[item] - have)

    progress = {"have_" + t: bool(st.inv.get(t)) for t in GOAL_TOOLS[goal]}
    progress.update({"crafting_table_placed": st.known_any("table"),
                     "crafting_table_within_reach": st.within_reach("table")})
    materials = ["wood", "stone"] + (["coal", "iron"] if goal == "iron_pickaxe" else [])
    progress.update({m: enough(m) for m in materials})
    if goal == "iron_pickaxe":
        progress.update({"furnace_placed": st.known_any("furnace"),
                         "table_and_furnace_both_within_reach":
                             st.within_reach("table") and st.within_reach("furnace"),
                         "have_sword": bool(st.inv.get("wood_sword") or st.inv.get("stone_sword"))})
    ox, oy = VIEW[0] // 2, VIEW[1] // 2
    lava_close = any(st.known.inside((st.pos[0] + dx, st.pos[1] + dy))
                     and st.known.mat[(st.pos[0] + dx, st.pos[1] + dy)] == "lava"
                     for dx in range(-ox, ox + 1) for dy in range(-oy, oy + 1))
    return {
        "goal": GOAL_TEXT[goal],
        "progress": progress,
        "inventory": inv or "empty",
        "vitals": {k: "%s (%d of 9)" % (_level(st.inv[k]), st.inv[k])
                   for k in ("health", "food", "drink", "energy")},
        "time_of_day": _time_words(st.daylight),
        "threats": threats or "none visible",
        "where_things_are": dict({"trees": how_far("tree"), "stone": how_far("stone"),
                                  "water": how_far("water"), "crafting table": how_far("table")},
                                 **({"coal": how_far("coal"), "iron": how_far("iron"),
                                     "furnace": how_far("furnace")} if goal == "iron_pickaxe" else {})),
        **({"lava_on_screen": True} if lava_close else {}),
        "map_knowledge": "only what has been seen so far" if st.known.mode == "explored"
                         else "the whole map",
        "last_step": last_outcome,
        **({"plan_from_planner": plan} if plan else {}),
        "rules": RULES + (RULES_IRON if goal == "iron_pickaxe" else ""),
    }


class FakeJev:
    """--dry: same request/response shape, no network, no cost."""

    backend = "dry"
    model = "fake"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.calls = 0

    def ask(self, state, questions):
        self.calls += 1
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": round(self.rng.random(), 2)}
            else:
                keys = list(q["criteria"])
                w = [self.rng.random() ** 3 for _ in keys]
                tot = sum(w)
                probs = {k: round(x / tot, 2) for k, x in zip(keys, w)}
                best = max(probs, key=probs.get)
                answers[qid] = {"type": "choice", "choice": best, "probabilities": probs,
                                "confidence": probs[best]}
        return {"answers": answers, "usage": {}, "_latency_s": 0.0, "model": "fake"}

    def stats(self):
        return {"backend": "dry", "calls": self.calls}


class JevAgent:
    name = "jev"
    level = "option"

    def __init__(self, goal, backend="typesafe", model=None, max_calls=150, dry=False, seed=0,
                 opus=None, escalate=0.0, use_plan=False):
        self.goal = goal
        self.client = FakeJev(seed) if dry else JevClient(backend=backend, model=model,
                                                          max_calls=max_calls)
        self.rng = random.Random(seed)
        self.script = ScriptedAgent(goal)     # for the agreement measurement only
        self.last_outcome = "just started"
        self.opus = opus                      # planner.Opus / FakeOpus, or None
        self.escalate = escalate              # hand to Opus when Jev's confidence < this
        self.use_plan = use_plan and opus is not None
        self.plan = None
        self.name = "jev-opus" if opus is not None else "jev"

    def prepare(self, st):
        """Once, before the first step: Opus writes the plan Jev will be shown."""
        if self.use_plan and self.plan is None:
            from planner import OpusError
            try:
                self.plan = self.opus.plan(GOAL_TEXT[self.goal], situation(st, self.goal, self.last_outcome))
            except OpusError:
                self.plan = []
        return self.plan

    def decide(self, st, options):
        scripted = self.script.decide(st, options).key
        if len(options) == 1:
            only = options[0]
            d = Decision(only.key, probs={only.key: 1.0}, source="only option",
                         note="nothing to choose between")
            d.extra = {"script_choice": scripted, "confidence": None}
            return d

        order = list(options)
        self.rng.shuffle(order)
        questions = {
            "next_step": {
                "type": "choice",
                "instructions": "Which option should the player take next, to make progress "
                                "toward the goal while staying alive?",
                "criteria": {o.key: o.label for o in order},
            },
            "in_danger": {
                "type": "noul",
                "instructions": "Is the player in immediate danger of losing health?",
                "criteria": {"true": "Something is about to hurt the player, or already is.",
                             "false": "Nothing threatens the player right now."},
            },
            "should_recover": {
                "type": "noul",
                "instructions": "Should the player drink, eat or rest soon rather than keep working?",
                "criteria": {"true": "A vital is low enough that recovering should come first soon.",
                             "false": "Vitals are fine; keep working on the goal."},
            },
        }
        sit = situation(st, self.goal, self.last_outcome, self.plan)
        data = self.client.ask(sit, questions)
        ans = data["answers"]
        ch = ans["next_step"]
        key = ch["choice"]
        if key not in {o.key for o in options}:          # defensive: never act on a non-option
            key = max(ch["probabilities"], key=ch["probabilities"].get)
        conf = ch.get("confidence")
        d = Decision(key, probs=ch.get("probabilities"), source="jev",
                     note="top choice at %d%%" % round(100 * ch["probabilities"].get(key, 0)))
        d.extra = {
            "confidence": conf,
            "danger": ans["in_danger"]["noul"],
            "recover": ans["should_recover"]["noul"],
            "latency_s": data.get("_latency_s"),
            "served_model": data.get("model"),
            "script_choice": scripted,
        }
        if self.opus is not None and self.escalate and conf is not None and conf < self.escalate:
            d = self._escalate(d, sit, options)
        return d

    def _escalate(self, d, sit, options):
        """Jev was unsure: ask Opus, from the same menu. Keep Jev's probabilities
        for the panel so the viewer can see what Jev was torn between."""
        from planner import OpusError
        d.extra["escalated"] = True
        d.extra["jev_choice"] = d.key
        d.extra["jev_confidence"] = d.extra["confidence"]
        try:
            r = self.opus.decide(sit, [(o.key, o.label) for o in options])
        except OpusError as e:
            d.extra["escalation_error"] = str(e)
            d.note += " (Opus unavailable - kept Jev's pick)"
            return d
        if r is None:
            d.extra["escalation_error"] = "no usable answer"
            d.note += " (Opus gave no usable answer - kept Jev's pick)"
            return d
        key, reason, secs = r
        out = Decision(key, probs=d.probs, source="opus", note=reason)
        out.extra = dict(d.extra, opus_seconds=round(secs, 2), overrode=(key != d.key))
        return out

    def outcome(self, text):
        self.last_outcome = text

    def stats(self):
        s = dict(self.client.stats())
        if self.opus is not None:
            s["opus"] = self.opus.stats()
        return s
