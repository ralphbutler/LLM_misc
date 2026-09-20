#!/usr/bin/env python3
"""Run the same worlds through several setups and print the comparison table.

Every results table in FINDINGS.md came from this: N worlds x M setups x R runs
each, in parallel, with one line per world and a summary at the end.

Measurement noise is the thing to watch. One run per world is close to a coin
toss - in a 40-world A/B of a single change, 8 worlds flipped and no world
failed in both arms. Use --runs 2 or more before believing a small difference.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))

# name -> the run.py flags that define it
SETUPS = {
    "random": ["--agent", "random"],
    "scripted": ["--agent", "scripted"],
    "jev": ["--agent", "jev"],
    "jev-opus-noplan": ["--agent", "jev-opus", "--no-opus-plan"],
    "jev-opus-plan": ["--agent", "jev-opus", "--escalate", "0"],
    "jev-opus": ["--agent", "jev-opus"],
}

EPILOG = """
setups (what each one means):
  random           key presses at random - the floor
  scripted         the hand-written recipe - the reference to beat
  jev              Jev picks every decision, no Opus
  jev-opus-noplan  Jev picks; Opus decides when Jev's confidence < 0.55
  jev-opus-plan    Opus writes a plan up front; NO hand-off (--escalate 0)
  jev-opus         both: the plan and the hand-off

examples:
  # free, no keys: is the script's night behaviour any good?
  uv run python compare.py --goal iron_pickaxe --setups scripted --runs 2

  # the FINDINGS.md table (~$4, 66 games) - the three Opus setups, twice each
  uv run python compare.py --goal iron_pickaxe \\
      --setups jev-opus-plan,jev-opus-noplan,jev-opus --runs 2

  # rehearse any of it for free
  uv run python compare.py --setups jev,jev-opus --dry

seeds: "21,1-10" is the 11-world set used throughout FINDINGS.md.
cost:  printed per setup at the end, and --dry always costs nothing.
"""


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="compare.py", epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run several setups over the same worlds and print the comparison table.")
    ap.add_argument("--setups", default="scripted,jev,jev-opus-noplan",
                    help="comma-separated, in table order (default: scripted,jev,jev-opus-noplan). "
                         "Choices: " + ", ".join(SETUPS))
    ap.add_argument("--seeds", default="21,1-10",
                    help="worlds, e.g. '21,1-10' or '1-40' (default: 21,1-10)")
    ap.add_argument("--runs", type=int, default=1,
                    help="runs per world per setup; 2+ before believing small gaps (default: 1)")
    ap.add_argument("--goal", default="iron_pickaxe", choices=["stone_pickaxe", "iron_pickaxe"],
                    help="what the agents are trying to make (default: iron_pickaxe)")
    ap.add_argument("--map", default="explored", choices=["full", "explored"],
                    help="what the agents may know (default: explored)")
    ap.add_argument("--jobs", type=int, default=4,
                    help="games in parallel. Keep it modest when models are called (default: 4)")
    ap.add_argument("--backend", default="typesafe", choices=["typesafe", "openrouter"],
                    help="where Jev is called (default: typesafe)")
    ap.add_argument("--tag", default="cmp", help="label for this comparison's logs (default: cmp)")
    ap.add_argument("--keep-logs", action="store_true",
                    help="keep every run's .jsonl decision log (default: delete after reading)")
    ap.add_argument("--out", default=None, help="also write the raw rows to this JSON file")
    ap.add_argument("--dry", action="store_true", help="fake models: free, no keys, tests the pipeline")
    args = ap.parse_args(argv)
    bad = [s for s in args.setups.split(",") if s not in SETUPS]
    if bad:
        ap.error("unknown setup(s): %s\nchoices: %s" % (", ".join(bad), ", ".join(SETUPS)))
    return args


def _one(job):
    """One game, in its own process. Returns the row, never raises."""
    setup, seed, rep, args = job
    import run
    tag = "%s_%s_r%d" % (args.tag, setup, rep)
    argv = SETUPS[setup] + ["--goal", args.goal, "--seed", str(seed), "--map", args.map,
                            "--no-video", "--tag", tag]
    if args.dry:
        argv += ["--dry"]
    elif setup.startswith("jev"):
        argv += ["--backend", args.backend]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run.main(argv)
    except Exception as exc:                      # a crashed game is data, not a stopped sweep
        return {"setup": setup, "seed": seed, "rep": rep, "error": repr(exc), "success": False}
    agent = SETUPS[setup][1]
    base = os.path.join(HERE, "results", "%s_seed%d_%s" % (agent, seed, tag))
    with open(base + "_summary.json") as f:
        s = json.load(f)
    if not args.keep_logs:
        for suffix in ("_summary.json", ".jsonl"):
            with contextlib.suppress(OSError):
                os.remove(base + suffix)
    opus = s.get("opus") or {}
    return {"setup": setup, "seed": seed, "rep": rep, "error": None,
            "success": s["success"], "steps": s["steps"], "cause": s.get("death_cause"),
            "health": s["inventory"].get("health"),
            "decisions": s["decisions"], "agreements": s["agreements"],
            "escalations": s.get("escalations"), "overrides": s.get("overrides"),
            "cost": (s.get("est_cost_usd") or 0) + (opus.get("cost_usd") or 0)}


def main(argv=None):
    args = parse_args(argv)
    setups = args.setups.split(",")
    seeds = parse_seeds(args.seeds)
    needs_jev = any(s.startswith("jev") for s in setups)
    needs_opus = any(s.startswith("jev-opus") for s in setups)
    if not args.dry:
        key = {"typesafe": "TYPESAFE_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[args.backend]
        if needs_jev and not os.environ.get(key):
            print("ERROR: %s is not set, so Jev cannot be called.\n"
                  "  - export it, or use --backend %s, or rehearse free with --dry"
                  % (key, "openrouter" if args.backend == "typesafe" else "typesafe"), file=sys.stderr)
            return 2
        if needs_opus and not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY is not set, so Opus cannot be called.\n"
                  "  - export it, drop the jev-opus setups, or rehearse free with --dry", file=sys.stderr)
            return 2

    jobs = [(setup, seed, rep, args)
            for setup in setups for rep in range(1, args.runs + 1) for seed in seeds]
    print("%d games: %d setups x %d worlds x %d run(s), %d at a time%s"
          % (len(jobs), len(setups), len(seeds), args.runs, args.jobs, "  [dry]" if args.dry else ""))
    rows, done = [], 0
    with Pool(args.jobs) as pool:
        for r in pool.imap_unordered(_one, jobs):
            rows.append(r)
            done += 1
            print("  %3d/%d  %-16s seed %-3d %s" % (
                done, len(jobs), r["setup"], r["seed"],
                "ok %4d steps" % r["steps"] if r["success"]
                else (r.get("error") or r.get("cause") or "out of steps")), flush=True)

    print("\n%-16s %-9s %-7s %-7s %-10s %s" % ("setup", "reached", "median", "health", "agreed", "cost"))
    for setup in setups:
        r = [x for x in rows if x["setup"] == setup]
        ok = [x for x in r if x["success"]]
        ag = sum(x.get("agreements") or 0 for x in r)
        de = sum(x.get("decisions") or 0 for x in r)
        print("%-16s %2d/%-6d %-7s %-7s %-10s $%.2f" % (
            setup, len(ok), len(r),
            "%.0f" % statistics.median([x["steps"] for x in ok]) if ok else "-",
            "%.1f" % statistics.mean([x["health"] for x in ok]) if ok else "-",
            ("%d/%d" % (ag, de)) if ag else "-",
            sum(x.get("cost") or 0 for x in r)))

    if args.runs > 1 or len(setups) > 1:
        print("\nper-world (o = reached, X = failed), worlds in order: %s" % args.seeds)
        for setup in setups:
            cells = []
            for seed in seeds:
                got = [x["success"] for x in rows if x["setup"] == setup and x["seed"] == seed]
                cells.append("".join("o" if v else "X" for v in got))
            print("  %-16s %s" % (setup, " ".join(cells)))

    fails = [x for x in rows if not x["success"]]
    if fails:
        print("\nfailures:")
        for x in sorted(fails, key=lambda z: (z["setup"], z["seed"])):
            print("  %-16s seed %-3d %s" % (x["setup"], x["seed"],
                                            x.get("error") or x.get("cause") or "out of steps"))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print("\nraw rows: %s" % args.out)
    print("\nOne run per world is close to a coin toss; treat gaps under ~3 in 22 as noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
