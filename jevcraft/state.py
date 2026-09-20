"""What an agent is allowed to know about the Crafter world, plus pathfinding.

KnownMap is the agent's knowledge of the map. Phase 1 fills it with the whole
map ("full"); the demo will fill it only with tiles that have been on screen
("explored"). Same object, one flag - decision 2 in PLAN.md.

Coordinates are Crafter's: world[x, y], x grows right, y grows down.
"""

from __future__ import annotations

import heapq

import numpy as np
from crafter import constants, objects

DIRS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
DIR_NAME = {v: k for k, v in DIRS.items()}
ACTION = {name: i for i, name in enumerate(constants.actions)}
WALKABLE = set(constants.walkable)            # grass, sand, path - NOT lava
VIEW = (9, 7)                                 # world tiles on screen (x, y)

# Solid tiles that `do` clears, and the tools each needs. A tree leaves grass,
# stone/coal/iron/diamond leave path, so all of them can be dug through.
CLEARABLE = {
    mat: dict(info["require"])
    for mat, info in constants.collect.items()
    if info["leaves"] in WALKABLE and mat not in WALKABLE
}
WALK_COST = 1
CLEAR_COST = 3        # turn, do, then step in


class KnownMap:
    def __init__(self, env, mode="full"):
        if mode not in ("full", "explored"):
            raise ValueError("mode must be 'full' or 'explored'")
        self.env = env
        self.mode = mode
        world = env._world
        self.area = tuple(world.area)
        n = max(world._mat_names) + 1
        self._names = np.array([world._mat_names.get(i) for i in range(n)], dtype=object)
        self.mat = np.full(self.area, None, dtype=object)   # None = never seen
        self.seen = np.zeros(self.area, bool)

    def update(self):
        world = self.env._world
        if self.mode == "full":
            self.mat = self._names[world._mat_map]
            self.seen[:] = True
            return
        px, py = self.env._player.pos
        ox, oy = VIEW[0] // 2, VIEW[1] // 2
        x0, x1 = max(0, px - ox), min(self.area[0], px - ox + VIEW[0])
        y0, y1 = max(0, py - oy), min(self.area[1], py - oy + VIEW[1])
        self.mat[x0:x1, y0:y1] = self._names[world._mat_map[x0:x1, y0:y1]]
        self.seen[x0:x1, y0:y1] = True

    def inside(self, t):
        return 0 <= t[0] < self.area[0] and 0 <= t[1] < self.area[1]

    def tiles_of(self, material):
        xs, ys = np.nonzero(self.mat == material)
        return list(zip(xs.tolist(), ys.tolist()))


class State:
    """One step's snapshot: everything an agent or the renderer reads."""

    def __init__(self, env, known, step):
        p = env._player
        self.env = env
        self.known = known
        self.step = step
        self.pos = (int(p.pos[0]), int(p.pos[1]))
        self.facing = tuple(int(v) for v in p.facing)
        self.inv = dict(p.inventory)
        self.sleeping = p.sleeping
        self.daylight = float(env._world.daylight)
        self.achievements = {k for k, v in p.achievements.items() if v > 0}
        # Creatures move, so in explored mode only the ones on screen right now
        # are known - a zombie seen ten steps ago is not a fact any more.
        ox, oy = VIEW[0] // 2, VIEW[1] // 2
        self.creatures = []
        for o in env._world.objects:
            if o is p:
                continue
            t = (int(o.pos[0]), int(o.pos[1]))
            on_screen = abs(t[0] - self.pos[0]) <= ox and abs(t[1] - self.pos[1]) <= oy
            if known.mode == "full" or on_screen:
                self.creatures.append((type(o).__name__.lower(), t))
        self.blocked = {t for _, t in self.creatures}

    # --- facts computed in code, so Jev never has to do arithmetic --------

    def has(self, requirements):
        return all(self.inv.get(k, 0) >= v for k, v in requirements.items())

    def within_reach(self, material):
        """Crafter's `nearby` test: the 3x3 square around the player."""
        x, y = self.pos
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                t = (x + dx, y + dy)
                if self.known.inside(t) and self.known.mat[t] == material:
                    return True
        return False

    def can_make(self, tool):
        info = constants.make[tool]
        return self.has(info["uses"]) and all(self.within_reach(u) for u in info["nearby"])

    def can_afford_place(self, thing):
        return self.has(constants.place[thing]["uses"])

    def adjacent(self, kind):
        """Creatures of `kind` in the 4 tiles next to the player."""
        x, y = self.pos
        return [t for k, t in self.creatures if k == kind and abs(t[0] - x) + abs(t[1] - y) == 1]

    def known_any(self, material):
        return bool((self.known.mat == material).any())

    def hostiles(self):
        """(manhattan distance, tile) for every visible zombie/skeleton, nearest first."""
        x, y = self.pos
        return sorted((abs(t[0] - x) + abs(t[1] - y), t) for k, t in self.creatures
                      if k in ("zombie", "skeleton"))

    # --- movement cost for pathfinding -------------------------------------

    def step_cost(self, t):
        if not self.known.inside(t) or t in self.blocked:
            return None
        mat = self.known.mat[t]
        if mat in WALKABLE:
            return WALK_COST
        req = CLEARABLE.get(mat)
        if req is not None and self.has(req):
            return CLEAR_COST
        return None                    # lava, water, table, unknown, un-diggable


def neighbors(t):
    return [(t[0] + d[0], t[1] + d[1]) for d in DIRS.values()]


def dijkstra(st, goal):
    """Cheapest route from the player to any tile where goal(tile) is true.
    Returns the list of tiles to step through (empty if already there), or None."""
    start = st.pos
    if goal(start):
        return []
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]
    while pq:
        c, u = heapq.heappop(pq)
        if c > dist[u]:
            continue
        if u != start and goal(u):
            path = [u]
            while prev[path[-1]] != start:
                path.append(prev[path[-1]])
            return path[::-1]
        for v in neighbors(u):
            w = st.step_cost(v)
            if w is None:
                continue
            nc = c + w
            if nc < dist.get(v, 1 << 30):
                dist[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc, v))
    return None


def dijkstra_all(st, limit=None):
    """Cost to reach every reachable tile, plus the predecessor map."""
    start = st.pos
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]
    while pq:
        c, u = heapq.heappop(pq)
        if c > dist[u]:
            continue
        if limit is not None and c >= limit:
            continue
        for v in neighbors(u):
            w = st.step_cost(v)
            if w is None:
                continue
            nc = c + w
            if nc < dist.get(v, 1 << 30):
                dist[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc, v))
    return dist, prev


def path_to(prev, start, goal):
    if goal == start:
        return []
    path = [goal]
    while prev[path[-1]] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def step_along(st, path):
    """Primitive action for the first tile of a path. Solid-but-clearable tiles
    get turned to, then dug; walking into a solid tile only turns you."""
    n = path[0]
    d = (n[0] - st.pos[0], n[1] - st.pos[1])
    if st.known.mat[n] in WALKABLE:
        return ACTION["move_" + DIR_NAME[d]]
    if st.facing == d:
        return ACTION["do"]
    return ACTION["move_" + DIR_NAME[d]]


def face_and_do(st, target):
    """Target must be 4-adjacent and solid or occupied, so moving toward it turns."""
    d = (target[0] - st.pos[0], target[1] - st.pos[1])
    if st.facing == d:
        return ACTION["do"]
    return ACTION["move_" + DIR_NAME[d]]
