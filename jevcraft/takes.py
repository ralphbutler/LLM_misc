#!/usr/bin/env python3
"""Record several takes, keep the best, and append an honest end card.

Decision 6 in PLAN.md: several takes, the best one goes in the video, and the
end card says so - "best of N runs on this seed; K/N succeeded" - plus one
baseline line (random / script / Jev) on the same seed. Every number on the
card comes from the logged runs; nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

EPILOG = """\
examples:
  uv run python takes.py --seed 21 --map explored            # 5 Jev+Opus takes + baselines -> final video
  uv run python takes.py --seed 21 --map explored --goal iron_pickaxe --no-opus-plan   # the iron video
  uv run python takes.py --seed 21 --map explored --agent jev   # Jev alone (the phase-2 video)
  uv run python takes.py --seed 21 --map explored --takes 8
  uv run python takes.py --seed 21 --dry --takes 2           # free rehearsal with fake Jev
  uv run python takes.py --seed 21 --map explored --pick 3   # use take 3, not the automatic best

how the best take is chosen (unless --pick):
  reached the goal first, then most health left, then fewest steps.

outputs:
  videos/takes/<variant>_seed<N>_take<i>.mp4   every take, no end card
  videos/final_<variant>_seed<N>.mp4           the chosen take + the end card
  (<variant> = jev, jev-opus, or jev-opus-noplan - so setups never overwrite each other)
  results/takes_seed<N>.json               every take's summary + baselines

cost: a Jev take is ~10 calls (~$0.0004). A jev-opus take adds 1-3 Opus calls
(~$0.01 each). Random/script baselines are free.
"""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="takes.py", description="Record N takes, pick the best, append an honest end card.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=21, help="world seed (default: 21)")
    ap.add_argument("--map", default="explored", choices=["full", "explored"],
                    help="what the agent may know (default: explored)")
    ap.add_argument("--goal", default="stone_pickaxe", choices=["stone_pickaxe", "iron_pickaxe"],
                    help="what the agent is trying to make; iron_pickaxe is the long run "
                         "(default: stone_pickaxe). Iron takes and end card are filmed at 8 fps")
    ap.add_argument("--agent", default="jev-opus", choices=["jev-opus", "jev"],
                    help="who is filmed: jev-opus = Jev + Opus hand-off; jev = Jev alone "
                         "(default: jev-opus)")
    ap.add_argument("--takes", type=int, default=5, help="how many filmed takes (default: 5)")
    ap.add_argument("--no-opus-plan", action="store_true",
                    help="with --agent jev-opus: hand-offs only, no up-front Opus plan "
                         "(in the 11-world comparison this was as fast and half the cost)")
    ap.add_argument("--jev-baseline", type=int, default=3,
                    help="with --agent jev-opus: also run Jev ALONE this many times, no video, "
                         "for the comparison line (default: 3; ~$0.0004 each)")
    ap.add_argument("--baselines", type=int, default=5,
                    help="runs each of random and scripted on the same seed, no video (default: 5)")
    ap.add_argument("--pick", type=int, default=None,
                    help="use this take number instead of the automatic best")
    ap.add_argument("--backend", default="typesafe", choices=["typesafe", "openrouter"],
                    help="where Jev is called (default: typesafe)")
    ap.add_argument("--minimap", default="slide", choices=["slide", "widen"],
                    help="minimap behaviour, passed to run.py (default: slide)")
    ap.add_argument("--dry", action="store_true", help="fake Jev: free rehearsal")
    return ap.parse_args(argv)


def _run(argv):
    import run
    with contextlib.redirect_stdout(io.StringIO()):
        code = run.main(argv)
    return code


def _n(k, word):
    """'1 call', '3 calls' - end cards are read by people."""
    return "%d %s%s" % (k, word, "" if k == 1 else "s")


def _summary(agent, seed, tag):
    with open(os.path.join(HERE, "results", "%s_seed%d_%s_summary.json" % (agent, seed, tag))) as f:
        return json.load(f)


def main(argv=None):
    args = parse_args(argv)
    if not args.dry and args.agent == "jev-opus" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set in this shell, so Opus cannot be called.\n"
              "  - export it, or use --agent jev (Jev alone), or rehearse with --dry", file=sys.stderr)
        return 2
    if not args.dry:
        need = {"typesafe": "TYPESAFE_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[args.backend]
        if not os.environ.get(need):
            print("ERROR: %s is not set in this shell, so Jev cannot be called.\n"
                  "  - export it (or re-source the file your shell loads keys from, or open a new terminal)\n"
                  "  - or use the other backend: --backend %s\n"
                  "  - or rehearse for free: --dry"
                  % (need, "openrouter" if args.backend == "typesafe" else "typesafe"), file=sys.stderr)
            return 2
    import imageio_ffmpeg
    import crafter
    from render import Renderer, VideoOut

    iron = args.goal == "iron_pickaxe"
    fps = 8 if iron else 6          # must match run.py's per-goal default, or the
                                    # end card and the take cannot be concatenated
    common = ["--seed", str(args.seed), "--map", args.map, "--minimap", args.minimap,
              "--goal", args.goal]
    # Variant name keeps different setups from overwriting each other's files.
    noplan = args.no_opus_plan and args.agent == "jev-opus"
    variant = (args.agent + ("-noplan" if noplan else "") + ("_iron" if iron else "")
               + ("_dry" if args.dry else ""))
    # Dry runs get their own prefix too: a dry rehearsal once wrote a FAKE Jev
    # run into the same log a real run was reading, and it reached the end card.
    tprefix = ("dry_" if args.dry else "") + ("iron_" if iron else "") + ("noplan_" if noplan else "")
    takes_dir = os.path.join(HERE, "videos", "takes")
    os.makedirs(takes_dir, exist_ok=True)

    print("baselines: %d random + %d scripted runs on seed %d (free)..."
          % (args.baselines, args.baselines, args.seed))
    base, base_n, base_steps = {}, {}, {}
    plan = [("random", args.baselines), ("scripted", args.baselines)]
    if args.agent == "jev-opus" and args.jev_baseline:
        plan.append(("jev", args.jev_baseline))
    for agent, n in plan:
        wins, steps = 0, []
        for i in range(n):
            tag = "%sbase%d" % (tprefix, i)
            extra = (["--dry"] if args.dry else ["--backend", args.backend]) if agent == "jev" else []
            _run(["--agent", agent, "--no-video", "--tag", tag] + common + extra)
            s = _summary(agent, args.seed, tag)
            wins += s["success"]
            if s["success"]:
                steps.append(s["steps"])
        base[agent], base_n[agent] = wins, n
        base_steps[agent] = sorted(steps)[len(steps) // 2] if steps else None
        print("  %-8s %d/%d reached the goal%s" % (agent, wins, n,
              ("  (median %d steps)" % base_steps[agent]) if steps else ""))

    takes = []
    for i in range(1, args.takes + 1):
        tag = "%stake%d" % (tprefix, i)
        out = os.path.join(takes_dir, "%s_seed%d_take%d.mp4" % (variant, args.seed, i))
        extra = ["--dry"] if args.dry else ["--backend", args.backend]
        extra += ["--no-opus-plan"] if (args.no_opus_plan and args.agent == "jev-opus") else []
        _run(["--agent", args.agent, "--tag", tag, "--out", out, "--no-end-card"] + common + extra)
        s = _summary(args.agent, args.seed, tag)
        s["take"] = i
        takes.append(s)
        o = s.get("opus") or {}
        print("  take %d: %-26s %3d steps  health %s  %2d Jev calls  agreed %s%s"
              % (i, s["outcome"], s["steps"], s["inventory"].get("health", 0), s["jev_calls"],
                 "%d/%d" % (s["agreements"], s["jev_decisions"]) if s["jev_decisions"] else "-",
                 ("  Opus %d calls (%d esc, %d overrode) $%.3f"
                  % (o.get("calls", 0), s.get("escalations", 0), s.get("overrides", 0),
                     o.get("cost_usd", 0))) if o else ""))

    wins = sum(t["success"] for t in takes)
    if args.pick:
        best = next(t for t in takes if t["take"] == args.pick)
    else:
        best = max(takes, key=lambda t: (t["success"], t["inventory"].get("health", 0), -t["steps"]))

    env = crafter.Env(seed=args.seed, size=(864, 864))
    env.reset()
    R = Renderer(env, 864, 576, args.agent, args.seed, args.goal.replace("_", " "))
    who = "Jev + Opus" if args.agent == "jev-opus" else "Jev"

    def med(agent):
        return (" (median %d steps)" % base_steps[agent]) if base_steps.get(agent) else ""
    compare = "same seed:  random %d/%d  ·  hand-written script %d/%d%s" % (
        base["random"], base_n["random"], base["scripted"], base_n["scripted"], med("scripted"))
    if "jev" in base:
        compare += "  ·  Jev alone %d/%d%s" % (base["jev"], base_n["jev"], med("jev"))
    won = sorted(tk["steps"] for tk in takes if tk["success"])
    compare += "  ·  %s %d/%d%s" % (who, wins, len(takes),
                                   (" (median %d steps)" % won[len(won) // 2]) if won else "")
    o = best.get("opus") or {}
    lines = [
        "this video is the best of %d runs on this seed: %d of %d reached the goal"
        % (len(takes), wins, len(takes)),
        compare,
        "this run: %d steps  ·  %d Jev calls ($%.4f)  ·  agreed with the script on %d of %d choices"
        % (best["steps"], best["jev_calls"], best["est_cost_usd"], best["agreements"],
           best["jev_decisions"]),
    ]
    if o:
        lines.append("Opus: %s (%s%s when Jev was unsure; changed %s)  ·  $%.3f  ·  %.0f s"
                     % (_n(o.get("calls", 0), "call"), "" if args.no_opus_plan else "a plan + ",
                        _n(best.get("escalations", 0), "hand-off"), _n(best.get("overrides", 0), "choice"),
                        o.get("cost_usd", 0), o.get("seconds", 0)))
    lines += [
        "map: %s  ·  Jev via %s%s" % ("only what it has seen" if args.map == "explored" else "whole map",
                                     "fake (dry run)" if args.dry else args.backend,
                                     "" if args.dry else "  ·  model %s" % ((best.get("client") or {}).get("model_served") or "?")),
    ]
    card_path = os.path.join(takes_dir, "endcard_%s_seed%d.mp4" % (variant, args.seed))
    v = VideoOut(card_path, fps)
    v.add(R.card("Goal reached" if best["success"] else "Goal not reached", lines,
                 sub=best["outcome"]), 6.0)
    v.close()

    final = os.path.join(HERE, "videos", "final_%s_seed%d.mp4" % (variant, args.seed))
    listing = os.path.join(takes_dir, "concat_%s.txt" % variant)
    with open(listing, "w") as f:
        f.write("file '%s'\nfile '%s'\n" % (best["video"], card_path))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", listing, "-c", "copy", final], check=True)

    record = {"agent": args.agent, "goal": args.goal, "seed": args.seed, "map": args.map, "dry": args.dry,
              "baselines": base, "baseline_median_steps": base_steps,
              "baseline_runs": args.baselines, "takes": takes, "chosen": best["take"],
              "end_card": lines, "final_video": final}
    with open(os.path.join(HERE, "results", "takes_%s_seed%d.json" % (variant, args.seed)), "w") as f:
        json.dump(record, f, indent=2)

    print("chosen take : %d (%s)" % (best["take"], "--pick" if args.pick else "automatic"))
    print("end card    :")
    for line in lines:
        print("   " + line)
    print("final video : %s" % final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
