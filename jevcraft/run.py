#!/usr/bin/env python3
"""jevcraft - Jev plays Crafter.

Runs one episode with the chosen agent and records it to MP4 with a side
panel that says what the agent is doing and why.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _time_words(daylight):
    return "day" if daylight > 0.7 else "evening" if daylight > 0.45 else "night"


COST_PER_CALL = 0.000036          # measured in jevsearch
INTERRUPT_SHELTER = False         # may a zombie interrupt a shelter being dug? (I-1 check)

EPILOG = """\
examples:
  uv run python run.py --agent jev-opus --seed 21 --map explored   # Jev decides, Opus when unsure
  uv run python run.py --goal iron_pickaxe --seed 21 --map explored # the long iron run (script)
  uv run python run.py --agent jev --seed 21                  # Jev decides (TypeSafe API)
  uv run python run.py --agent jev --seed 21 --map explored   # ...and only knows what it has seen
  uv run python run.py --agent jev --backend openrouter       # same model, via OpenRouter
  uv run python run.py --agent jev --dry                      # fake Jev: free, checks the plumbing
  uv run python run.py --seed 21                              # the hand-written script
  uv run python run.py --agent random                         # the floor: no plan at all
  uv run python run.py --agent jev --live --no-video          # just watch, write nothing
  uv run python seeds.py                                      # find demo-friendly seeds

what you see in the video:
  left   the game; the camera follows the player. A YELLOW OUTLINE marks the
         tile the agent is working toward. Colored BANNERS across the top call
         out events: red = danger or failure (zombie adjacent, health lost),
         amber = caution (zombie approaching, Jev unsure), green = found something.
  right  NOW: what it is doing, in words. DECISION: every option on the menu,
         the one taken (>), Jev's probability for each, and why. Below that:
         danger / recover / confidence (colored when they matter), and a MAGENTA
         line whenever Jev chose differently from the hand-written script.
         INVENTORY. A MINIMAP: red = player, white trail = where it has been,
         white box = what the camera shows, yellow = the target, dark = unseen.
  Decisions are held on screen for --hold seconds so they can be read.

notes:
  The MP4 is the deliverable. It is written at a fixed --fps, so playback is
  smooth no matter how long Jev takes to decide.

  --live opens a pygame window for debugging - spotting a stuck or looping agent
  in seconds instead of after a render. It shows the same frames the MP4 gets.
  Close the window or press Esc/q to stop early; the MP4 is still written.

  --map explored: the agent only knows tiles that have been on screen, and
  "explore" joins the menu. This is the honest mode for the demo.

  --seed picks the WORLD (map layout). The same seed always builds the same
  map. Runs are NOT exactly reproducible, though, even for the random agent:
  Crafter keeps creatures in Python sets, whose order depends on memory
  addresses, so zombies and cows spawn and move a little differently each run.
  Jev adds its own non-determinism on top.

  jev-opus: Jev (fast, ~0.2 s, ~$0.00004 a call) makes every decision. When its
  top choice is under --escalate confidence, Opus (slow, ~5 s, ~$0.01 a call)
  decides instead, from the same menu - shown in BLUE in the video. Opus also
  writes a short plan once, up front, that Jev sees on every call.

  API keys: TYPESAFE_API_KEY for --backend typesafe (default),
  OPENROUTER_API_KEY for --backend openrouter, ANTHROPIC_API_KEY for Opus.
  --dry needs none of them.

outputs:
  videos/<agent>_seed<N>.mp4           the video
  results/<agent>_seed<N>.jsonl        one line per decision: menu, choice, probabilities, state
  results/<agent>_seed<N>_summary.json outcome, steps, decisions, calls, cost, agreement
"""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="run.py", description="Run a Crafter agent and record it to MP4.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="scripted", choices=["jev-opus", "jev", "scripted", "random"],
                    help="who decides: jev-opus = Jev picks, Opus plans and takes over when Jev "
                         "is unsure; jev = Jev alone; scripted = hand-written recipe (the "
                         "'perfect play' reference); random = no plan (the floor) (default: scripted)")
    ap.add_argument("--goal", default="stone_pickaxe", choices=["stone_pickaxe", "iron_pickaxe"],
                    help="what the agent is trying to make. iron_pickaxe is the long run: coal, "
                         "iron dug out of mountains, a furnace, and usually a night "
                         "(default: stone_pickaxe)")
    ap.add_argument("--seed", type=int, default=1,
                    help="world seed; same seed = same map (default: 1)")
    ap.add_argument("--steps", type=int, default=None,
                    help="max steps; the episode also ends on death or success "
                         "(default: 400 for stone, 900 for iron)")
    ap.add_argument("--map", default="full", choices=["full", "explored"],
                    help="what the agent may know: 'full' = whole map; 'explored' = only "
                         "tiles it has seen, with 'explore' on the menu (default: full)")
    ap.add_argument("--backend", default="typesafe", choices=["typesafe", "openrouter"],
                    help="where Jev is called: TypeSafe's own API or OpenRouter (default: typesafe)")
    ap.add_argument("--model", default=None,
                    help="Jev model name (default: jev-latest, or ~typesafe/jev-latest on OpenRouter)")
    ap.add_argument("--max-calls", type=int, default=150,
                    help="hard cap on Jev calls this episode, a spend guard (default: 150)")
    ap.add_argument("--dry", action="store_true",
                    help="use a fake Jev (and fake Opus) that answer randomly - free, for testing")
    ap.add_argument("--escalate", type=float, default=None,
                    help="hand a decision to Opus when Jev's confidence is below this "
                         "(default: 0.55 for jev-opus, off for jev)")
    ap.add_argument("--opus-effort", default="medium", choices=["low", "medium", "high"],
                    help="Opus effort: higher = slower and pricier, more careful (default: medium)")
    ap.add_argument("--opus-max-calls", type=int, default=20,
                    help="hard cap on Opus calls this episode, a spend guard (default: 20)")
    ap.add_argument("--no-opus-plan", action="store_true",
                    help="jev-opus without the up-front Opus plan (escalation only)")
    ap.add_argument("--minimap", default="slide", choices=["slide", "widen"],
                    help="explored-map minimap: 'slide' = 40x40 window that slides a tile at a "
                         "time near its edge (no jumps); 'widen' = 40x40 window that switches "
                         "to the whole world once when reached (default: slide)")
    ap.add_argument("--size", type=int, default=864,
                    help="game view in pixels; keep a multiple of 144 (default: 864)")
    ap.add_argument("--panel", type=int, default=576,
                    help="side panel width in pixels (default: 576)")
    ap.add_argument("--fps", type=int, default=None,
                    help="video frame rate; lower is easier to follow "
                         "(default: 6 for stone, 8 for the longer iron run)")
    ap.add_argument("--hold", type=float, default=1.2,
                    help="seconds each decision stays on screen (default: 1.2)")
    ap.add_argument("--out", default=None,
                    help="MP4 path (default: videos/<agent>_seed<seed>.mp4)")
    ap.add_argument("--tag", default="",
                    help="extra label for output names, e.g. take2 -> jev_seed21_take2")
    ap.add_argument("--no-video", action="store_true", help="don't write an MP4")
    ap.add_argument("--no-end-card", action="store_true",
                    help="skip the end card (takes.py adds its own, with best-of-N counts)")
    ap.add_argument("--live", action="store_true",
                    help="DEBUG: show a live pygame window while the agent plays")
    ap.add_argument("--live-scale", type=float, default=0.8,
                    help="size of the --live window relative to the video (default: 0.8)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    iron = args.goal == "iron_pickaxe"
    if args.steps is None:
        args.steps = 900 if iron else 400
    if args.fps is None:
        args.fps = 8 if iron else 6

    # Heavy imports only after argument parsing, so -h works anywhere.
    import crafter
    from crafter import constants
    from agents import RandomAgent, ScriptedAgent
    from render import LiveWindow, Renderer, VideoOut
    from skills import GOALS, goal_reached, menu
    from state import ACTION, KnownMap, State

    name = "%s_seed%d%s" % (args.agent, args.seed, ("_" + args.tag) if args.tag else "")
    if args.agent in ("jev", "jev-opus"):
        from chooser import JevAgent, JevError
        escalate = args.escalate if args.escalate is not None else (0.55 if args.agent == "jev-opus" else 0.0)
        opus = None
        if args.agent == "jev-opus" or escalate > 0:
            from planner import FakeOpus, Opus
            opus = FakeOpus(args.seed) if args.dry else Opus(args.opus_effort, args.opus_max_calls)
        try:
            agent = JevAgent(args.goal, backend=args.backend, model=args.model,
                             max_calls=args.max_calls, dry=args.dry, seed=args.seed,
                             opus=opus, escalate=escalate,
                             use_plan=(args.agent == "jev-opus" and not args.no_opus_plan))
        except JevError as e:
            print("ERROR: %s\n  set the key in this shell, try --backend %s, or rehearse with --dry"
                  % (e, "openrouter" if args.backend == "typesafe" else "typesafe"), file=sys.stderr)
            return 2
    else:
        class JevError(Exception):     # nothing to catch for non-Jev agents
            pass
        agent = RandomAgent(args.seed) if args.agent == "random" else ScriptedAgent(args.goal)

    env = crafter.Env(seed=args.seed, size=(args.size, args.size))
    obs = env.reset()
    known = KnownMap(env, args.map)
    known.update()
    st = State(env, known, 0)

    goal_label = GOALS[args.goal]
    label = agent.name + (" (dry)" if args.dry else "")
    R = Renderer(env, args.size, args.panel, label, args.seed, goal_label, minimap=args.minimap)

    video = out = None
    if not args.no_video:
        out = args.out or os.path.join(HERE, "videos", name + ".mp4")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        video = VideoOut(out, args.fps)
    live = LiveWindow(R.W, R.H, args.fps, "jevcraft - %s - seed %d" % (label, args.seed),
                      args.live_scale) if args.live else None

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    log_path = os.path.join(HERE, "results", name + ".jsonl")
    log = open(log_path, "w")

    def emit(frame, seconds=None):
        if video:
            video.add(frame, seconds)
        if live and not live.show(frame, seconds):
            raise KeyboardInterrupt

    plan = agent.prepare(st) if hasattr(agent, "prepare") else None
    who = {"jev-opus": "Jev picks every step; Opus plans, and decides when Jev is unsure",
           "jev": "Jev picks every step from a menu the code builds",
           "scripted": "a hand-written recipe picks every step",
           "random": "random key presses - no plan"}[agent.name]
    title_lines = [who, "%s map  ·  seed %d" % (args.map, args.seed),
                   "the panel on the right says what it is doing and why"]
    if plan:
        title_lines.append("")
        title_lines.append("Opus's plan:")
        title_lines.extend("%d.  %s" % (i, s) for i, s in enumerate(plan, 1))
    title = R.card("jevcraft", title_lines, sub="Goal: make a %s" % goal_label)

    trail = collections.deque(maxlen=90)
    trail.append(st.pos)
    alerts = []                       # [text, color, steps_left]
    skill = None
    hud_decision = None
    n_dec = failures = jev_decisions = agreements = 0
    escalations = overrides = 0
    last_interrupt = -99
    night_warned = -1
    banned = {}                       # option key -> step it may return
    success = died = stopped = False
    death_cause = None
    error = None
    t0 = time.monotonic()
    step = 0

    def alert(text, color, ttl=6):
        for a in alerts:
            if a[0] == text:
                a[2] = max(a[2], ttl)
                return
        alerts.append([text, color, ttl])

    try:
        emit(title, 3.0)
        for step in range(1, args.steps + 1):
            deciding = False
            hs = st.hostiles()
            if hs and hs[0][0] == 1:
                alert("ZOMBIE ADJACENT", "red", 2)
            elif hs and hs[0][0] <= 3:
                alert("zombie approaching", "amber", 2)

            if agent.level == "primitive":
                action = agent.act(st)
                intent = "Random action: " + constants.actions[action].replace("_", " ")
                reason, target = "no plan - this is the baseline to beat", None
            else:
                if (skill is not None and skill.status == "running" and hs and hs[0][0] == 1
                        and (skill.key not in ("fight_zombie", "flee_zombie")
                             and (INTERRUPT_SHELTER or skill.key != "shelter"))
                        and step - last_interrupt >= 8):
                    skill.status = "interrupted"
                    last_interrupt = step
                # Night is falling: stop whatever long skill is running and
                # reconsider, once per night - otherwise a 30-step explore can
                # carry the agent past dusk before sheltering is even on the menu.
                night_now = st.daylight < 0.6 and 130 <= step % 300 <= 250
                if (iron and night_now and night_warned != step // 300
                        and skill is not None and skill.status == "running"
                        and skill.key != "shelter"):
                    skill.status = "interrupted"
                    night_warned = step // 300
                    alert("night is falling", "blue", 6)
                action = None
                for _ in range(4):
                    if skill is None or skill.status != "running":
                        if skill is not None:
                            if skill.status == "failed":
                                failures += 1
                                banned[skill.key] = step + 10
                                alert("could not: " + skill.reason, "red", 10)
                            if getattr(skill, "found", None):
                                alert("found %s!" % skill.found, "green", 10)
                            if hasattr(agent, "outcome"):
                                agent.outcome("%s: %s%s" % (skill.key, skill.status,
                                              (" (" + skill.reason + ")") if skill.status == "failed" else ""))
                        opts = menu(st, args.goal, {k for k, until in banned.items() if until > step})
                        if not opts:
                            break
                        dec = agent.decide(st, opts)
                        chosen = next(o for o in opts if o.key == dec.key)
                        skill = chosen.make()
                        n_dec += 1
                        deciding = True
                        extra = dec.extra or {}
                        if dec.source in ("jev", "opus"):
                            jev_decisions += 1
                            agreements += int(extra.get("script_choice") == dec.key)
                            conf = extra.get("confidence")
                            if extra.get("escalated"):
                                escalations += 1
                                if dec.source == "opus":
                                    alert("Jev unsure (%d%%): asking Opus" % round(100 * conf), "blue", 5)
                                    if extra.get("overrode"):
                                        overrides += 1
                                        alert("Opus overrode Jev", "blue", 5)
                            elif conf is not None and conf < 0.55:
                                alert("Jev unsure: %d%%" % round(100 * conf), "amber", 4)
                        hud_decision = {"n": n_dec, "source": dec.source, "chosen": dec.key,
                                        "options": [(o.key, o.label) for o in opts],
                                        "probs": dec.probs, "note": dec.note, "extra": extra}
                        log.write(json.dumps({
                            "step": st.step, "decision": n_dec, "source": dec.source,
                            "options": [[o.key, o.label] for o in opts], "chosen": dec.key,
                            "probs": dec.probs, "note": dec.note, "extra": extra,
                            "pos": st.pos, "inventory": {k: v for k, v in st.inv.items() if v},
                            "daylight": round(st.daylight, 2)}) + "\n")
                    action = skill.step(st)
                    if action is not None:
                        break
                if action is None:
                    action = ACTION["noop"]
                intent = skill.intent if skill else "Idle"
                reason = skill.reason if skill else ""
                target = skill.target if skill else None

            hud = {"intent": intent, "reason": reason, "target": target,
                   "decision": hud_decision, "deciding": deciding, "trail": list(trail),
                   "alerts": [(a[0], a[1]) for a in alerts]}
            if deciding:
                emit(R.compose(obs, st, hud), args.hold)

            health_before = st.inv["health"]
            obs, reward, done, info = env.step(action)
            known.update()
            st = State(env, known, step)
            trail.append(st.pos)
            lost = health_before - st.inv["health"]
            if lost > 0:
                alert("-%d health" % lost, "red", 5)
            for a in alerts:
                a[2] -= 1
            alerts[:] = [a for a in alerts if a[2] > 0]
            hud.update(deciding=False, trail=list(trail), alerts=[(a[0], a[1]) for a in alerts])
            emit(R.compose(obs, st, hud))

            if agent.level == "option" and goal_reached(st, args.goal):
                success = True
                hud.update(intent="Goal reached: %s" % goal_label, reason="", target=None,
                           alerts=[("GOAL REACHED", "green")])
                emit(R.compose(obs, st, hud), 2.5)
                break
            if done:
                died = info.get("discount", 1) == 0
                if died:
                    here = env._world[env._player.pos][0]
                    empty = [k for k in ("food", "drink", "energy") if st.inv.get(k, 0) == 0]
                    near = [d for d, _ in st.hostiles() if d <= 2]
                    death_cause = ("lava" if here == "lava" else
                                   "zombie/skeleton" if near and not empty else
                                   ("no " + "/".join(empty)) + (" + zombie" if near else "") if empty
                                   else "unknown")
                    death_cause += " (%s, step %d)" % (_time_words(st.daylight), step)
                break
    except KeyboardInterrupt:
        stopped = True
    except JevError as e:
        error = str(e)

    wall = time.monotonic() - t0
    outcome = ("reached the %s" % goal_label if success else "died" if died else
               "api error" if error else "stopped early" if stopped else "ran out of steps")
    calls = agent.stats().get("calls", 0) if hasattr(agent, "stats") else 0
    opus_stats = (agent.stats().get("opus") if hasattr(agent, "stats") else None)
    paid_calls = 0 if args.dry else calls
    agree = (agreements / jev_decisions) if jev_decisions else None
    if not stopped and not args.no_end_card:
        lines = ["%s  ·  seed %d  ·  %d steps  ·  %d decisions" % (label, args.seed, step, n_dec)]
        if agent.name in ("jev", "jev-opus"):
            lines.append("%d Jev calls  ·  about $%.4f  ·  agreed with the script on %d of %d choices"
                         % (calls, paid_calls * COST_PER_CALL, agreements, jev_decisions))
        if opus_stats:
            lines.append("Opus: %d calls (plan + %d escalations, changed %d choices)  ·  about $%.3f  ·  %.0f s"
                         % (opus_stats["calls"], escalations, overrides, opus_stats["cost_usd"],
                            opus_stats.get("seconds", 0)))
        lines.append("achievements: %s" % (", ".join(a.replace("_", " ") for a in sorted(st.achievements)) or "none"))
        end = R.card("Goal reached" if success else "Goal not reached", lines, sub=outcome)
        try:
            emit(end, 4.0)
        except KeyboardInterrupt:
            pass
    if video:
        video.close()
    if live:
        live.close()
    log.close()

    summary = {"agent": agent.name, "dry": args.dry, "seed": args.seed, "map": args.map,
               "goal": args.goal, "tag": args.tag, "outcome": outcome, "success": success,
               "died": died, "death_cause": death_cause, "error": error, "steps": step, "decisions": n_dec,
               "jev_decisions": jev_decisions, "agreements": agreements, "agreement": agree,
               "jev_calls": calls, "est_cost_usd": round(paid_calls * COST_PER_CALL, 5),
               "escalations": escalations, "overrides": overrides, "plan": plan,
               "opus": opus_stats,
               "skill_failures": failures, "achievements": sorted(st.achievements),
               "inventory": {k: v for k, v in st.inv.items() if v}, "wall_s": round(wall, 1),
               "backend": args.backend if agent.name in ("jev", "jev-opus") else None,
               "client": agent.stats() if hasattr(agent, "stats") else None, "video": out}
    with open(os.path.join(HERE, "results", name + "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("agent        : %s" % label)
    print("seed / map   : %d / %s" % (args.seed, args.map))
    print("outcome      : %s%s" % (outcome, ("  (" + error + ")") if error else
                                    ("  (" + death_cause + ")") if death_cause else ""))
    print("steps        : %d" % step)
    print("decisions    : %d  (skill failures: %d)" % (n_dec, failures))
    if opus_stats:
        print("opus         : %d calls  ~$%.3f  %.0f s   escalations %d, overrides %d"
              % (opus_stats["calls"], opus_stats["cost_usd"], opus_stats.get("seconds", 0),
                 escalations, overrides))
    if plan:
        print("opus plan    : %s" % "  ->  ".join(plan))
    if agent.name in ("jev", "jev-opus"):
        print("jev calls    : %d  (~$%.4f)   agreed with script: %s"
              % (calls, paid_calls * COST_PER_CALL,
                 "%d/%d" % (agreements, jev_decisions) if jev_decisions else "n/a"))
    print("achievements : %s" % (", ".join(sorted(st.achievements)) or "none"))
    print("wall seconds : %.1f" % wall)
    if out:
        print("video        : %s  (%.1fs)" % (out, video.frames / args.fps))
    print("decision log : %s" % log_path)
    return 0 if not error else 2


if __name__ == "__main__":
    sys.exit(main())
