#!/usr/bin/env python3
"""Scan Crafter seeds for demo-friendly worlds, offline, no API.

The map is fully determined by the seed (creature behavior is not - see
run.py -h), so a seed can be judged by its map alone: how far to trees, to
stone, to water, how much lava is close by, and whether coal and iron are
reachable for the later iron-pickaxe run.
"""

from __future__ import annotations

import argparse
import sys

EPILOG = """\
examples:
  uv run python seeds.py                    # score seeds 1-50, show the best 10
  uv run python seeds.py --count 200 --top 20
  uv run python seeds.py --iron             # rank for the iron-pickaxe run instead

columns (all in steps from the start, '-' = unreachable):
  tree   walk to the first tree          trees10  trees within 10 steps
  stone  walk to the first stone         water    walk to water
  coal   dig to coal (with a pickaxe)    iron     dig to iron (with a stone pickaxe)
  lava12 lava tiles within 12 steps - fewer is safer
  score  lower is better for the chosen goal
"""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="seeds.py", description="Rank Crafter seeds by how demo-friendly their map is.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1, help="first seed to scan (default: 1)")
    ap.add_argument("--count", type=int, default=50, help="how many seeds to scan (default: 50)")
    ap.add_argument("--top", type=int, default=10, help="how many to print (default: 10)")
    ap.add_argument("--iron", action="store_true",
                    help="rank for the iron-pickaxe run (needs coal + iron) instead of stone")
    return ap.parse_args(argv)


def measure(seed):
    import crafter
    from state import KnownMap, State, dijkstra, neighbors

    env = crafter.Env(seed=seed)
    env.reset()
    known = KnownMap(env, "full")
    known.update()
    st = State(env, known, 0)
    sx, sy = st.pos

    def reach(material, tools=()):
        for t in tools:
            st.inv[t] = 1
        path = dijkstra(st, lambda u: st.known.mat[u] in ("grass", "sand", "path") and any(
            known.inside(n) and known.mat[n] == material for n in neighbors(u)))
        for t in tools:
            st.inv[t] = 0
        return None if path is None else len(path)

    def within(material, r):
        return sum(1 for (x, y) in known.tiles_of(material) if abs(x - sx) + abs(y - sy) <= r)

    return {
        "seed": seed,
        "tree": reach("tree"),
        "trees10": within("tree", 10),
        "stone": reach("stone"),
        "water": reach("water"),
        "coal": reach("coal", ("wood_pickaxe",)),
        "iron": reach("iron", ("wood_pickaxe", "stone_pickaxe")),
        "lava12": within("lava", 12),
    }


def score(m, iron):
    big = 999
    s = (m["tree"] if m["tree"] is not None else big) + (m["stone"] if m["stone"] is not None else big)
    s += 5 * max(0, 4 - m["trees10"]) + 3 * m["lava12"]
    if iron:
        s += (m["coal"] if m["coal"] is not None else big) + (m["iron"] if m["iron"] is not None else big)
        s += m["water"] if m["water"] is not None else big
    return s


def main(argv=None):
    args = parse_args(argv)
    rows = []
    for seed in range(args.start, args.start + args.count):
        m = measure(seed)
        m["score"] = score(m, args.iron)
        rows.append(m)
    rows.sort(key=lambda m: m["score"])
    cols = ["seed", "tree", "trees10", "stone", "water", "coal", "iron", "lava12", "score"]
    print("ranked for the %s run  (%d seeds scanned)" % ("IRON" if args.iron else "STONE", len(rows)))
    print("  ".join("%7s" % c for c in cols))
    for m in rows[:args.top]:
        print("  ".join("%7s" % ("-" if m[c] is None else m[c]) for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
