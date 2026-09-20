"""Agents.

Two kinds:
  primitive - picks one of Crafter's 17 raw actions every step (random)
  option    - picks from menu() between skills, via decide(state, options)

The option interface is the one Jev will use in phase 2: it receives the
feasible options with all numbers precomputed, and returns a key plus,
optionally, probabilities for the side panel.
"""

from __future__ import annotations

import numpy as np
from crafter import constants

from skills import remaining


class Decision:
    def __init__(self, key, probs=None, source="", note=""):
        self.key = key
        self.probs = probs          # {option_key: p} or None
        self.source = source        # "scripted", "jev", "opus"
        self.note = note            # one line of why, for the panel
        self.extra = {}             # model-specific details (confidence, latency, ...)


class RandomAgent:
    """Uniform over the 17 primitive actions. The floor every agent must beat."""

    name = "random"
    level = "primitive"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, st):
        return int(self.rng.integers(len(constants.actions)))


class ScriptedAgent:
    """Fixed recipe for the goal, written by hand. The 'perfect play' reference -
    the oracle's role from jevsearch. It sees exactly the menu Jev will see."""

    name = "scripted"
    level = "option"

    # I-1 survival behaviours for the iron run, each switchable so the
    # measurement can add them one at a time. The stone run uses none of them.
    swords = True        # craft a sword from spare materials
    shelter = True       # dig in and wall off at night
    eat_early = True     # drink/eat at 3 rather than 2
    shelter_first = True # night beats food/fights (False = the old order)

    def __init__(self, goal):
        self.goal = goal

    def decide(self, st, options):
        keys = {o.key for o in options}
        iron = self.goal == "iron_pickaxe"

        def pick(key, note):
            return Decision(key, source="scripted", note=note)

        low = 3 if (iron and self.eat_early) else 2

        # Night first. "shelter" is only on the menu while night is coming or
        # here and a rock pocket is in reach, and that beats everything else
        # the agent could be doing: the measurement showed runs spending the
        # whole night hunting a cow that was not there, with stone in hand and
        # "shelter" on the menu the entire time (seeds 30, 38). The only thing
        # that comes first is an adjacent zombie when health is too low to take
        # the hits on the way in; the shelter skill deals with one in the
        # doorway itself.
        if (iron and self.shelter and self.shelter_first and "shelter" in keys
                and "make_iron_pickaxe" not in keys):
            if "fight_zombie" in keys and st.inv["health"] <= 4:
                return pick("fight_zombie", "too hurt to dig in with a zombie on me")
            if st.inv["drink"] < 7 and "drink_water" in keys:
                return pick("drink_water", "top up before a long night in the shelter")
            if st.inv["food"] < 6 and "eat_cow" in keys:
                return pick("eat_cow", "eat before a long night in the shelter")
            return pick("shelter", "night: wall in with stone until morning")

        if "fight_zombie" in keys:
            return pick("fight_zombie", "a zombie is adjacent - deal with it first")
        if st.inv["drink"] <= low and "drink_water" in keys:
            return pick("drink_water", "thirst is getting critical")
        if st.inv["food"] <= low and "eat_cow" in keys:
            return pick("eat_cow", "hunger is getting critical")
        if iron and self.eat_early:
            if st.inv["food"] <= 5 and "eat_cow" in keys:
                return pick("eat_cow", "a cow is in sight and food is half gone")
            if st.inv["food"] <= 4 and "find_cow" in keys:
                return pick("find_cow", "food is low and no cow in sight")
        if (iron and self.shelter and not self.shelter_first and "shelter" in keys
                and "make_iron_pickaxe" not in keys):
            if st.inv["drink"] < 7 and "drink_water" in keys:
                return pick("drink_water", "top up before a long night in the shelter")
            if st.inv["food"] < 6 and "eat_cow" in keys:
                return pick("eat_cow", "eat before a long night in the shelter")
            return pick("shelter", "night: wall in with stone until morning")
        if iron and self.swords:
            for sword in ("make_stone_sword", "make_wood_sword"):
                if sword in keys:
                    return pick(sword, "spare materials; zombies die in fewer hits")

        if "make_iron_pickaxe" in keys:
            return pick("make_iron_pickaxe", "everything needed is in hand")
        if "make_stone_pickaxe" in keys:
            return pick("make_stone_pickaxe", "everything needed is in hand"
                        if self.goal == "stone_pickaxe" else "iron can only be mined with it")
        need = remaining(st, self.goal)
        if iron and self.swords and not (st.inv.get("wood_sword") or st.inv.get("stone_sword")):
            need = dict(need, wood=need["wood"] + 1, stone=need["stone"] + 1)
        if self.goal == "iron_pickaxe" and st.inv.get("stone_pickaxe"):
            return self._iron_phase(st, keys, need, pick) or pick(
                options[0].key, "fallback: first feasible option")
        wood, stone = st.inv.get("wood", 0), st.inv.get("stone", 0)
        have_table = st.known_any("table")

        if st.inv.get("wood_pickaxe", 0):
            if stone < need["stone"]:
                if "collect_stone" in keys:
                    return pick("collect_stone", "the stone pickaxe needs stone")
                if "explore" in keys:
                    return pick("explore", "need stone and none has been seen yet")
            if "go_to_table" in keys:
                return pick("go_to_table", "have the stone; crafting needs the table")
        else:
            if "make_wood_pickaxe" in keys:
                return pick("make_wood_pickaxe", "stone can only be mined with a pickaxe")
            if wood >= need["wood"]:
                if not have_table and "place_table" in keys:
                    return pick("place_table", "enough wood for the table and both pickaxes")
                if have_table and "go_to_table" in keys:
                    return pick("go_to_table", "the wood pickaxe is made at the table")
        if "collect_wood" in keys:
            return pick("collect_wood", "%d more wood needed" % max(1, need["wood"] - wood))
        if "explore" in keys:
            return pick("explore", "need wood and no tree has been seen yet")
        return pick(options[0].key, "fallback: first feasible option") if options else None


    def _iron_phase(self, st, keys, need, pick):
        """After the stone pickaxe: gather coal, iron and whatever else is short,
        then bring table and furnace together. The script always walks back to
        the existing table - the 'second table' option is left for Jev/Opus."""
        for item, key in (("coal", "collect_coal"), ("iron", "collect_iron"),
                          ("stone", "collect_stone"), ("wood", "collect_wood")):
            if st.inv.get(item, 0) < need.get(item, 0):
                if key in keys:
                    return pick(key, "the iron pickaxe still needs %s" % item)
                if "explore" in keys:
                    return pick("explore", "need %s and none is reachable yet" % item)
        if "go_to_crafting_spot" in keys:
            return pick("go_to_crafting_spot", "table and furnace are placed; crafting needs both")
        if "place_furnace" in keys and st.within_reach("table"):
            return pick("place_furnace", "materials in hand; put the furnace beside the table")
        if "go_to_table" in keys:
            return pick("go_to_table", "walk back so the furnace can go beside the table")
        if "place_furnace" in keys:
            return pick("place_furnace", "no table to walk back to; place the furnace here")
        return None


AGENTS = {"random": RandomAgent, "scripted": ScriptedAgent}
