"""Skills: one chosen option -> a stream of primitive Crafter actions.

A skill replans every step (cows and zombies wander into paths), exposes a
`target` tile and an `intent` sentence for the renderer, and ends as "done" or
"failed". The agent is only consulted between skills - which is where Jev will
sit in phase 2.

menu() lists the options that are feasible right now, with every number
already worked out in code. That list is exactly what Jev will be shown.
"""

from __future__ import annotations

import collections

from crafter import constants

from state import (ACTION, CLEARABLE, WALKABLE, DIRS, DIR_NAME, dijkstra, dijkstra_all,
                   face_and_do, neighbors, path_to, step_along)

STUCK_LIMIT = 12       # steps without moving or gaining anything
STEP_LIMIT = 250       # hard cap on any single skill


def standable(st, t):
    return st.known.inside(t) and st.known.mat[t] in WALKABLE and t not in st.blocked


def approachable(st, t):
    """Can end up standing here: open ground, or a tile we can dig into with the
    tools held. Iron and coal sit inside stone with no open ground beside them,
    so 'standable only' would call almost all of it unreachable."""
    if standable(st, t):
        return True
    if not st.known.inside(t) or t in st.blocked:
        return False
    req = CLEARABLE.get(st.known.mat[t])
    return req is not None and st.has(req)


class Skill:
    key = "skill"

    def __init__(self):
        self.status = "running"
        self.target = None
        self.intent = ""
        self.reason = ""
        self.steps = 0
        self._last = None
        self._still = 0

    def step(self, st):
        """Return a primitive action index, or None once done/failed."""
        if self.status != "running":
            return None
        self.steps += 1
        sig = (st.pos, tuple(sorted(st.inv.items())))
        self._still = self._still + 1 if sig == self._last else 0
        self._last = sig
        if self.steps > STEP_LIMIT or self._still > STUCK_LIMIT:
            return self._fail("stuck")
        return self._step(st)

    def _step(self, st):
        raise NotImplementedError

    def _done(self):
        self.status = "done"
        return None

    def _fail(self, why):
        self.status = "failed"
        self.reason = why
        return None


class Collect(Skill):
    """Walk to the nearest `material`, `do` it until we hold `until` of it."""

    def __init__(self, material, until):
        super().__init__()
        self.material = material
        self.item = next(iter(constants.collect[material]["receive"]))
        self.until = until
        self.key = "collect_" + material

    def _step(self, st):
        have = st.inv.get(self.item, 0)
        self.intent = {"tree": "Chopping wood", "stone": "Mining stone", "coal": "Mining coal",
                       "iron": "Mining iron", "water": "Drinking water"}.get(self.material,
                                                                             "Collecting " + self.item)
        self.reason = "have %d of %d" % (have, self.until)
        if have >= self.until:
            return self._done()
        adj = [t for t in neighbors(st.pos) if st.known.inside(t) and st.known.mat[t] == self.material]
        if adj:
            ahead = (st.pos[0] + st.facing[0], st.pos[1] + st.facing[1])
            self.target = ahead if ahead in adj else adj[0]
            return face_and_do(st, self.target)
        path = dijkstra(st, lambda u: approachable(st, u) and any(
            st.known.inside(n) and st.known.mat[n] == self.material for n in neighbors(u)))
        if path is None:
            return self._fail("no reachable %s" % self.material)
        end = path[-1]
        self.target = next(n for n in neighbors(end)
                           if st.known.inside(n) and st.known.mat[n] == self.material)
        self.intent = "Heading to the nearest %s" % self.material
        return step_along(st, path)


def _near_all(u, tile_sets):
    return all(any((u[0] + dx, u[1] + dy) in ts for dx in (-1, 0, 1) for dy in (-1, 0, 1))
               for ts in tile_sets)


class GoNear(Skill):
    """Walk until every material in `material` is inside the 3x3 square around
    us (crafting reach). One material ("table") or several ("table", "furnace")."""

    def __init__(self, material, key=None):
        super().__init__()
        self.materials = (material,) if isinstance(material, str) else tuple(material)
        self.material = self.materials[0]
        self.key = key or ("go_to_" + self.material)

    def _step(self, st):
        names = " and ".join(self.materials)
        self.intent = ("Walking back to the %s" % names if len(self.materials) == 1
                       else "Walking to the %s" % names)
        self.reason = "crafting needs %s within one tile" % ("it" if len(self.materials) == 1 else "both")
        if all(st.within_reach(m) for m in self.materials):
            return self._done()
        sets = [set(st.known.tiles_of(m)) for m in self.materials]
        if not all(sets):
            return self._fail("no known %s" % names)
        path = dijkstra(st, lambda u: standable(st, u) and _near_all(u, sets))
        if path is None:
            return self._fail("cannot reach %s" % names)
        end = path[-1] if path else st.pos
        self.target = min(sets[0], key=lambda t: abs(t[0] - end[0]) + abs(t[1] - end[1]))
        return step_along(st, path)


class Place(Skill):
    """Put a table/furnace down. Crafter places on the tile you FACE, and the
    only way to face open ground is to have just walked toward it - so we walk
    to P, step once more in direction d onto S, and place on T = S + d."""

    def __init__(self, thing, keep_near=None):
        super().__init__()
        self.thing = thing
        self.keep_near = keep_near     # e.g. "table": place the furnace beside it
        self.key = "place_" + thing
        self.where = set(constants.place[thing]["where"])
        self._start = None
        self._plan = None          # (P, d)

    def _placeable(self, st, t):
        return st.known.inside(t) and st.known.mat[t] in self.where and t not in st.blocked

    def _step(self, st):
        self.intent = "Placing a %s" % self.thing
        self.reason = "costs %s" % ", ".join("%d %s" % (v, k) for k, v in constants.place[self.thing]["uses"].items())
        if self._start is None:
            self._start = len(st.known.tiles_of(self.thing))
        if len(st.known.tiles_of(self.thing)) > self._start:
            return self._done()
        if not st.can_afford_place(self.thing):
            return self._fail("cannot afford")
        keep = set(st.known.tiles_of(self.keep_near)) if self.keep_near else None
        ahead = (st.pos[0] + st.facing[0], st.pos[1] + st.facing[1])
        if self._placeable(st, ahead) and (keep is None or _near_all(st.pos, [keep])):
            self.target = ahead
            return ACTION["place_" + self.thing]

        def ok(u):
            if not standable(st, u):
                return False
            for d in DIRS.values():
                s = (u[0] + d[0], u[1] + d[1])
                t = (s[0] + d[0], s[1] + d[1])
                if (standable(st, s) and self._placeable(st, t)
                        and (keep is None or _near_all(s, [keep]))):
                    self._plan = (u, d)
                    return True
            return False
        if self._plan and self._plan[0] == st.pos:
            d = self._plan[1]
            self._plan = None
            return ACTION["move_" + {v: k for k, v in DIRS.items()}[d]]
        self._plan = None
        path = dijkstra(st, ok)
        if path is None or self._plan is None:
            return self._fail("no room to place")
        P, d = self._plan
        self.target = (P[0] + 2 * d[0], P[1] + 2 * d[1])
        if not path:              # already standing on P
            self._plan = None
            return ACTION["move_" + {v: k for k, v in DIRS.items()}[d]]
        return step_along(st, path)


class Make(Skill):
    def __init__(self, tool):
        super().__init__()
        self.tool = tool
        self.key = "make_" + tool
        self._start = None

    def _step(self, st):
        self.intent = "Crafting a %s" % self.tool.replace("_", " ")
        self.reason = "uses %s" % ", ".join("%d %s" % (v, k) for k, v in constants.make[self.tool]["uses"].items())
        if self._start is None:
            self._start = st.inv.get(self.tool, 0)
        if st.inv.get(self.tool, 0) > self._start:
            return self._done()
        if not st.can_make(self.tool):
            return self._fail("requirements not met")
        return ACTION["make_" + self.tool]


class Attack(Skill):
    """Hit an adjacent creature until it is gone (zombie) or eaten (cow)."""

    def __init__(self, kind, chase=False):
        super().__init__()
        self.kind = kind
        self.chase = chase
        self.key = ("eat_" if kind == "cow" else "fight_") + kind
        self._food0 = None

    def _step(self, st):
        self.intent = "Eating a cow" if self.kind == "cow" else "Fighting a %s" % self.kind
        if self.kind == "cow":
            if self._food0 is None:
                self._food0 = st.inv["food"]
            self.reason = "food %d" % st.inv["food"]
            if st.inv["food"] > self._food0:
                return self._done()
        adj = st.adjacent(self.kind)
        if adj:
            self.target = adj[0]
            return face_and_do(st, adj[0])
        if not self.chase:
            return self._done()          # the zombie left or died
        herd = [t for k, t in st.creatures if k == self.kind]
        if not herd:
            return self._fail("no %s in sight" % self.kind)
        path = dijkstra(st, lambda u: standable(st, u) and any(n in herd for n in neighbors(u)))
        if path is None:
            return self._fail("cannot reach a %s" % self.kind)
        end = path[-1] if path else st.pos
        self.target = min(herd, key=lambda t: abs(t[0] - end[0]) + abs(t[1] - end[1]))
        self.intent = "Chasing a %s" % self.kind
        return step_along(st, path) if path else ACTION["noop"]


class Sleep(Skill):
    key = "sleep"

    def _step(self, st):
        self.intent = "Sleeping"
        self.reason = "energy %d" % st.inv["energy"]
        if self.steps > 1 and not st.sleeping:
            return self._done()
        return ACTION["sleep"]


class Flee(Skill):
    """Step to whichever free neighbor is farthest from every visible hostile.
    A zombie follows, but only hits when adjacent - so moving away buys time."""

    key = "flee_zombie"

    def __init__(self, max_steps=10, safe=5):
        super().__init__()
        self.max_steps = max_steps
        self.safe = safe

    def _step(self, st):
        hs = st.hostiles()
        self.intent = "Backing away from a zombie"
        if not hs or hs[0][0] >= self.safe or self.steps > self.max_steps:
            return self._done()
        self.reason = "nearest zombie %d step%s away" % (hs[0][0], "" if hs[0][0] == 1 else "s")
        self.target = hs[0][1]
        best, best_d = None, -1
        for n in neighbors(st.pos):
            if not standable(st, n):
                continue
            d = min(abs(n[0] - t[0]) + abs(n[1] - t[1]) for _, t in hs)
            if d > best_d:
                best, best_d = n, d
        if best is None:
            return self._fail("cornered")
        return ACTION["move_" + DIR_NAME[(best[0] - st.pos[0], best[1] - st.pos[1])]]


COMPASS = {(1, 0): "east", (-1, 0): "west", (0, -1): "north", (0, 1): "south"}


class Explore(Skill):
    """Walk into unseen land along one heading, until something we are looking
    for comes into view or the step budget runs out. Only meaningful with the
    explored-only map."""

    key = "explore"

    def __init__(self, looking_for=(), max_steps=30, creature=None, key=None):
        super().__init__()
        self.looking_for = tuple(looking_for)
        self.creature = creature          # e.g. "cow": stop as soon as one is on screen
        if key:
            self.key = key
        self.max_steps = max_steps
        self.heading = None
        self.found = None

    def _pick_heading(self, st):
        k = st.known
        px, py = st.pos
        best, best_n = None, -1
        for d in COMPASS:
            n = 0
            for a in range(4, 17):
                for b in range(-8, 9):
                    t = (px + d[0] * a + (b if d[0] == 0 else 0), py + d[1] * a + (b if d[1] == 0 else 0))
                    if k.inside(t) and not k.seen[t]:
                        n += 1
            if n > best_n:
                best, best_n = d, n
        return best

    def _step(self, st):
        for m in self.looking_for:
            if reachable(st, m):
                self.found = m
                return self._done()
        if self.creature and any(k == self.creature for k, _ in st.creatures):
            self.found = "a " + self.creature
            return self._done()
        if self.steps > self.max_steps:
            return self._done()
        if self.heading is None:
            self.heading = self._pick_heading(st)
        k = st.known
        dist, prev = dijkstra_all(st, limit=80)
        frontier = [t for t in dist if approachable(st, t) and any(
            k.inside(n) and not k.seen[n] for n in neighbors(t))]
        if not frontier:
            return self._fail("nothing unexplored within reach")
        hx, hy = self.heading
        best = max(frontier, key=lambda t: 2 * ((t[0] - st.pos[0]) * hx + (t[1] - st.pos[1]) * hy) - dist[t])
        self.target = best
        self.intent = "Exploring %s" % COMPASS[self.heading]
        self.reason = ("looking for a %s" % self.creature if self.creature else
                       ("looking for %s" % " and ".join(self.looking_for)) if self.looking_for
                       else "mapping new land")
        path = path_to(prev, st.pos, best)
        if not path:
            return self._done()
        return step_along(st, path)


def _solid(st, t):
    """Known, and nothing can walk on it (zombies walk only grass/sand/path)."""
    return st.known.inside(t) and st.known.mat[t] is not None and st.known.mat[t] not in WALKABLE


def _open_or_diggable(st, t):
    if not st.known.inside(t) or t in st.blocked:
        return False
    mat = st.known.mat[t]
    if mat in WALKABLE:
        return True
    req = CLEARABLE.get(mat)
    return req is not None and st.has(req)


def find_shelter_site(st, limit=40):
    """A spot to dig a sealed two-tile room: entrance E, then A and B in a
    straight line, rock on every side of A and B and beyond B. Returns
    (E, d) for the nearest one, or None.

    Why two tiles: Crafter only lets you place a block on the tile you face,
    and you can only face open ground by stepping onto it. Dig A and B, step
    into B, step BACK into A - now you face E and can wall it off."""
    if not st.inv.get("stone"):
        return None
    dist, _ = dijkstra_all(st, limit=limit)
    best = None
    for e, c in dist.items():
        if not standable(st, e) and e != st.pos:
            continue
        for d in DIRS.values():
            a = (e[0] + d[0], e[1] + d[1])
            b = (a[0] + d[0], a[1] + d[1])
            beyond = (b[0] + d[0], b[1] + d[1])
            side = (d[1], d[0])
            walls = [(a[0] + side[0], a[1] + side[1]), (a[0] - side[0], a[1] - side[1]),
                     (b[0] + side[0], b[1] + side[1]), (b[0] - side[0], b[1] - side[1]), beyond]
            if (_open_or_diggable(st, a) and _open_or_diggable(st, b)
                    and all(_solid(st, w) for w in walls)
                    and (best is None or c < best[0])):
                best = (c, e, d)
    return None if best is None else (best[1], best[2])


class Shelter(Skill):
    """Dig a two-tile room into rock, wall off the entrance, sleep/wait until
    morning. Zombies spawn on grass and cannot dig."""

    key = "shelter"

    def __init__(self, wake_light=0.6):
        super().__init__()
        self.wake_light = wake_light
        self.site = None
        self.sealed = False

    def _step(self, st):
        if self.site is None:
            self.site = find_shelter_site(st)
            if self.site is None:
                return self._fail("no spot to dig a shelter")
        e, d = self.site
        a = (e[0] + d[0], e[1] + d[1])
        b = (a[0] + d[0], a[1] + d[1])
        back = (-d[0], -d[1])
        self.target = a
        mat = lambda t: st.known.mat[t]

        if self.sealed or (st.pos in (a, b) and _solid(st, e)):
            self.sealed = True
            self._still = 0                       # waiting in place is the point
            self.intent = "Sheltering until morning"
            self.reason = "walled in with stone; zombies cannot reach"
            if st.daylight > self.wake_light and not st.sleeping:
                return self._done()
            if st.inv["energy"] < 9 and not st.sleeping:
                return ACTION["sleep"]
            return ACTION["noop"]

        self.intent = "Digging a shelter for the night"
        self.reason = "night brings many zombies; rock on every side"
        if st.pos == b:                           # step back into A, now facing E
            return ACTION["move_" + DIR_NAME[back]]
        if st.pos == a:
            if st.facing == back:
                if e in st.blocked:               # a zombie in the doorway: hit it, then seal
                    self.reason = "a zombie is in the doorway - fighting it off before sealing"
                    return ACTION["do"]
                return ACTION["place_stone"] if not _solid(st, e) else ACTION["noop"]
            if mat(b) in WALKABLE:
                return ACTION["move_" + DIR_NAME[d]]
            return ACTION["do"] if st.facing == d else ACTION["move_" + DIR_NAME[d]]
        if st.pos == e:
            if mat(a) in WALKABLE:
                return ACTION["move_" + DIR_NAME[d]]
            return ACTION["do"] if st.facing == d else ACTION["move_" + DIR_NAME[d]]
        path = dijkstra(st, lambda u: u == e)
        if path is None:
            return self._fail("cannot reach the shelter spot")
        return step_along(st, path)


def zombie_hits(st):
    """Hits to kill a zombie with the best weapon held - Crafter's own damage table."""
    dmg = max(1, 2 if st.inv.get("wood_sword") else 0, 3 if st.inv.get("stone_sword") else 0,
              5 if st.inv.get("iron_sword") else 0)
    return -(-5 // dmg)


# --- goals ------------------------------------------------------------------

# item -> the material you mine for it (wood -> tree, stone -> stone, coal -> coal, ...)
MATERIAL_OF = {next(iter(info["receive"])): mat for mat, info in constants.collect.items()
               if mat not in ("water", "grass")}


def remaining(st, goal):
    """Raw materials STILL to gather for `goal`, given what is already held or
    placed - read from Crafter's own recipe tables, and computed here so Jev
    never has to subtract (the jug-puzzle lesson).

    Walks the recipe: a tool needs its ingredients, the utilities it must be
    crafted beside (table, furnace), and the tools needed to MINE its
    ingredients (stone needs a wood pickaxe, iron a stone pickaxe). Each tool
    and utility is counted once; anything already held or placed costs nothing."""
    need = collections.defaultdict(int)
    tools, utils = set(), set()

    def tools_to_mine(item):
        mat = MATERIAL_OF.get(item)
        if mat:
            for tool in constants.collect[mat]["require"]:
                need_tool(tool)

    def need_util(u):
        if u in utils or st.known_any(u):
            return
        utils.add(u)
        for k, v in constants.place[u]["uses"].items():
            need[k] += v
            tools_to_mine(k)

    def need_tool(t):
        if t in tools or st.inv.get(t, 0) >= 1:
            return
        tools.add(t)
        info = constants.make[t]
        for k, v in info["uses"].items():
            need[k] += v
            tools_to_mine(k)
        for u in info["nearby"]:
            need_util(u)

    if goal not in GOALS:
        raise ValueError("unknown goal %r" % goal)
    need_tool(goal)
    need["wood"] += 0
    need["stone"] += 0
    return need


GOALS = {"stone_pickaxe": "stone pickaxe", "iron_pickaxe": "iron pickaxe"}
GOAL_TOOLS = {"stone_pickaxe": ("wood_pickaxe", "stone_pickaxe"),
              "iron_pickaxe": ("wood_pickaxe", "stone_pickaxe", "iron_pickaxe")}


def goal_reached(st, goal):
    return st.inv.get(goal, 0) >= 1


# --- the menu: feasible options, every number precomputed ---------------------

class Option:
    def __init__(self, key, label, factory):
        self.key = key
        self.label = label
        self.factory = factory

    def make(self):
        return self.factory()


def reachable(st, material):
    """Seen AND there is a known route to stand next to it - walking, or digging
    with the tools held (iron and coal are usually walled in by stone)."""
    return path_len_to(st, material) is not None


def path_len_to(st, material):
    if not st.known_any(material):
        return None
    path = dijkstra(st, lambda u: approachable(st, u) and any(
        st.known.inside(n) and st.known.mat[n] == material for n in neighbors(u)))
    return None if path is None else len(path)


def distance_words(n):
    return ("not reachable" if n is None else "right here" if n <= 1 else "close"
            if n <= 5 else "a short walk away" if n <= 12 else "far away")


def crafting_spot_reachable(st):
    """Is there a place to stand with the table AND the furnace both in reach?"""
    sets = [set(st.known.tiles_of("table")), set(st.known.tiles_of("furnace"))]
    if not all(sets):
        return False
    return dijkstra(st, lambda u: standable(st, u) and _near_all(u, sets)) is not None


def _have_need(have, need):
    return "have %d, %s" % (have, "%d more needed" % (need - have) if need > have else "enough for the goal")


def menu(st, goal, banned=()):
    """Feasible options now. `banned` = keys that just failed, rested for a while
    so no chooser - script or Jev - can loop on an option that cannot work."""
    need = remaining(st, goal)
    wood, stone = st.inv.get("wood", 0), st.inv.get("stone", 0)
    have_table = st.known_any("table")
    opts = []

    hs = st.hostiles()
    if st.adjacent("zombie"):
        opts.append(Option("fight_zombie", "fight the zombie next to you (%d hits to kill; "
                           "usually costs 2-4 health)" % zombie_hits(st), lambda: Attack("zombie")))
    if hs and hs[0][0] <= 2:
        opts.append(Option("flee_zombie", "back away from the zombie (%s; it follows "
                           "and keeps pace, so it will be back)"
                           % ("it is adjacent" if hs[0][0] == 1 else "it is 2 steps away"),
                           lambda: Flee()))
    if reachable(st, "tree"):
        until = max(need["wood"], wood + 1)
        opts.append(Option("collect_wood", "collect wood (%s)" % _have_need(wood, need["wood"]),
                           lambda u=until: Collect("tree", u)))
    table_cost = constants.place["table"]["uses"].get("wood", 0)
    if st.can_afford_place("table"):
        if goal == "stone_pickaxe":
            opts.append(Option("place_table", "place a crafting table here (%s)"
                               % ("already have one; this would be a second" if have_table
                                  else "both pickaxes are crafted at a table"),
                               lambda: Place("table")))
        elif not have_table:
            opts.append(Option("place_table", "place a crafting table here (all three pickaxes "
                               "are crafted at a table)", lambda: Place("table")))
        elif (not st.inv.get("iron_pickaxe") and not st.within_reach("table")
              and wood - table_cost >= need["wood"]):
            opts.append(Option("place_table", "place a second crafting table here (costs %d wood, "
                               "enough is left; the first table is %s; the iron pickaxe needs a "
                               "table and the furnace side by side)"
                               % (table_cost, distance_words(path_len_to(st, "table"))),
                               lambda: Place("table")))
    if have_table and not st.within_reach("table"):
        opts.append(Option("go_to_table", "walk back to the crafting table (crafting needs it "
                           "within one tile)", lambda: GoNear("table")))
    purpose = ({"wood_pickaxe": "lets you mine stone", "stone_pickaxe": "the goal"}
               if goal == "stone_pickaxe" else
               {"wood_pickaxe": "lets you mine stone and coal", "stone_pickaxe": "lets you mine iron",
                "iron_pickaxe": "the goal"})
    for tool in GOAL_TOOLS[goal]:
        if st.inv.get(tool, 0) == 0 and st.can_make(tool):
            what = purpose[tool]
            opts.append(Option("make_" + tool, "craft a %s (%s)" % (tool.replace("_", " "), what),
                               lambda t=tool: Make(t)))
    if st.has(constants.collect["stone"]["require"]) and reachable(st, "stone"):
        until = max(need["stone"], stone + 1)
        opts.append(Option("collect_stone", "mine stone (%s)" % _have_need(stone, need["stone"]),
                           lambda u=until: Collect("stone", u)))
    if goal == "iron_pickaxe" and not st.inv.get("iron_pickaxe"):
        for mat, verb in (("coal", "mine coal"), ("iron", "mine iron")):
            have = st.inv.get(mat, 0)
            if (need[mat] > have and st.has(constants.collect[mat]["require"])
                    and reachable(st, mat)):
                opts.append(Option("collect_" + mat, "%s (%s)" % (verb, _have_need(have, need[mat])),
                                   lambda m=mat, u=need[mat]: Collect(m, u)))
        has_spot = crafting_spot_reachable(st)
        if st.can_afford_place("furnace") and not has_spot:
            if st.within_reach("table"):
                opts.append(Option("place_furnace", "place the furnace here, beside the table (the "
                                   "iron pickaxe needs both within one tile)",
                                   lambda: Place("furnace", keep_near="table")))
            else:
                opts.append(Option("place_furnace", "place the furnace here (the table is %s; the "
                                   "iron pickaxe needs both side by side)"
                                   % distance_words(path_len_to(st, "table")),
                                   lambda: Place("furnace")))
        if has_spot and not (st.within_reach("table") and st.within_reach("furnace")):
            opts.append(Option("go_to_crafting_spot", "walk to the table and furnace (the iron "
                               "pickaxe needs both within one tile)",
                               lambda: GoNear(("table", "furnace"), key="go_to_crafting_spot")))
    if goal == "iron_pickaxe":
        best = ("stone_sword" if st.inv.get("stone_sword") else
                "wood_sword" if st.inv.get("wood_sword") else None)
        for sword in ("stone_sword", "wood_sword"):
            if best == "stone_sword" or (best == "wood_sword" and sword == "wood_sword"):
                continue
            uses = constants.make[sword]["uses"]
            spare_ok = all(st.inv.get(k, 0) - need[k] >= v for k, v in uses.items())
            if st.can_make(sword) and spare_ok:
                hits = {"wood_sword": 3, "stone_sword": 2}[sword]
                opts.append(Option("make_" + sword, "craft a %s (zombies then take %d hits instead "
                                   "of %d; uses only spare materials)"
                                   % (sword.replace("_", " "), hits, zombie_hits(st)),
                                   lambda t=sword: Make(t)))
        if (st.daylight < 0.6 and 130 <= st.step % 300 <= 250
                and find_shelter_site(st) is not None):
            opts.append(Option("shelter", "dig into the rock and wall yourself in until morning "
                               "(%s; zombies swarm at night)"
                               % ("night now" if st.daylight < 0.45 else "night is coming"),
                               lambda: Shelter()))
    if st.inv["drink"] < constants.items["drink"]["max"] and reachable(st, "water"):
        opts.append(Option("drink_water", "drink water (thirst %d/9)" % st.inv["drink"],
                           lambda: Collect("water", constants.items["drink"]["max"])))
    if st.inv["food"] < constants.items["food"]["max"] and any(k == "cow" for k, _ in st.creatures):
        opts.append(Option("eat_cow", "hunt and eat a cow (food %d/9)" % st.inv["food"],
                           lambda: Attack("cow", chase=True)))
    if st.inv["energy"] <= 3:
        opts.append(Option("sleep", "sleep (energy %d/9)" % st.inv["energy"], lambda: Sleep()))
    if (goal == "iron_pickaxe" and st.inv["food"] <= 4
            and not any(k == "cow" for k, _ in st.creatures)):
        opts.append(Option("find_cow", "search for a cow to eat (food %d/9, none in sight; "
                           "hunger at 0 drains health)" % st.inv["food"],
                           lambda: Explore(creature="cow", max_steps=25, key="find_cow")))
    if st.known.mode == "explored":
        wanted, why = [], []
        for m, item in (("tree", "wood"), ("stone", "stone"), ("coal", "coal"), ("iron", "iron")):
            if item in ("coal", "iron") and not st.has(constants.collect[m]["require"]):
                continue
            if need[item] > st.inv.get(item, 0) and not reachable(st, m):
                wanted.append(m)
                why.append("%s %s" % (m, "seen but no path to it yet" if st.known_any(m)
                                      else "not seen yet"))
        label = ("explore unseen land (%s)" % "; ".join(why) if why
                 else "explore unseen land (all materials the goal needs are already reachable)")
        opts.append(Option("explore", label, lambda w=tuple(wanted): Explore(w)))
    return [o for o in opts if o.key not in banned] or opts
