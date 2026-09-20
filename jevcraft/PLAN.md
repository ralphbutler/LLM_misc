# 🎮 jevcraft — Jev plays Crafter

**Goal:** a short video, recorded on the Mac, of Jev making the moment-to-moment
decisions that take a Crafter agent from bare hands to a **stone pickaxe**, with
the **iron pickaxe** as a stretch goal. A side panel beside the game shows what Jev
was asked and how it answered, so the viewer sees a model deciding rather than a
sprite wandering around.

**Lineage:** this is the `jevsearch` harness pointed at a game instead of a
puzzle. Our code owns the world and the legal moves, Jev judges between them, and
Opus sets the goals. Everything we learned in `jevsearch/FINDINGS.md` shapes the
design below.

---

## 🔎 Verified facts about Crafter

Read directly from the `crafter` 1.8.3 source, not from memory.

| fact | consequence for us |
|---|---|
| Pure Python. Needs `numpy imageio pillow opensimplex ruamel.yaml`; `pygame` only for the live GUI | `pip install crafter`, nothing else heavy |
| 17 primitive actions: move ×4, `do`, `sleep`, `place_*`, `make_*`, `noop` | too low-level for Jev, so we need a skill layer |
| `do` acts on the **facing** tile; moving into a solid tile turns without moving | a skill needs a "face the target, then `do`" pattern |
| `make_*` needs the table (and furnace) **within 1 tile**, diagonals included | stand next to the table; put table and furnace side by side |
| `info['semantic']` is the **whole 64×64 map** with creatures overlaid | pathfinding is easy, but it's omniscient, so the demo uses explored-only (decision 2) |
| Walking into **lava is instant death** | lava is never a legal path tile |
| Stone, coal and iron sit in mountains; stone is mineable with a wood pickaxe and leaves a walkable path | the pathfinder can tunnel through stone at a cost |
| Iron is patchy inside mountains; diamond is rare | iron is reachable on many seeds, not all, so seeds need scanning |
| Day/night cycle is 300 steps: dusk around step 150, dark around 180–240 | zombies spawn in the dark and the frames go dark on video |
| Drink drops every ~20 steps, food every ~25, energy every ~30, all starting at 9 | a stone-pickaxe run (<150 steps) never needs to eat or drink; an iron run does |
| Built-in `Recorder` writes MP4 via `imageio` | we write our own recorder anyway, to add the side panel |

**Resource bills**, computed from `data.yaml`:

| goal | raw materials | tools needed along the way |
|---|---|---|
| stone pickaxe | 4 wood, 1 stone | table, wood pickaxe |
| iron pickaxe | 5 wood, 5 stone, 1 coal, 1 iron | table, furnace, wood + stone pickaxe |

---

## 🏗️ Architecture — three layers

```
  Opus (slow, rare)      sets the goal, orders the subgoals, writes Jev's questions
        │                gets called again only when Jev is unsure (see ⚖️)
        ▼
  Jev (fast, per decision)  "which of these options best advances the subgoal?"
        │                   plus a couple of yes/no safety checks
        ▼
  Skills (our code)      go_to / collect / place / make / drink / flee / sleep
        │                compile each chosen option into primitive actions
        ▼
  Crafter env            steps, renders frames
```

### Skills layer — deterministic, no model

Each macro turns into primitive actions and runs until it finishes or gets
interrupted:

- `go_to(material)`: BFS over a **known-map object**, reused from
  `jevsearch/puzzles.py`. Phase 1 fills it with the full map; the demo fills it
  only with tiles seen so far, and "explore" joins Jev's menu.
  Grass, sand and path are free to walk; stone costs extra because it has to be
  mined (only once we hold a wood pickaxe); lava and water are forbidden.
- `collect(material)`: go_to, face the tile, `do` until the inventory count goes up.
- `place(table|furnace)`: pick a legal adjacent tile and face it.
- `make(tool)`, `drink()`, `eat_cow()`, `sleep()`, `flee()`, `fight()`

**Only feasible options reach Jev.** Code filters the menu, the same way the
puzzle harness generated only legal moves. Jev never sees "make wood pickaxe" when
there's no table within one tile.

### Jev layer — the chooser

Called when a skill finishes or an interrupt fires (a zombie next to us, health
dropping, night falling). It is **not** called on every step. That keeps it to
roughly 20–60 calls per episode, and it builds in commitment: Experiment E showed
~0.05 jitter on close calls, which would make a per-step loop flip back and forth.

One call per decision, with all questions in parallel:

| question | type | purpose |
|---|---|---|
| `next_step` | choice over the feasible skills (≤ 8, shuffled per call) | the actual decision |
| `in_danger` | noul | override: flee or fight before anything else |
| `should_recover` | noul | drink, eat or sleep now rather than later |

**The state sent to Jev is structured, and all arithmetic is done in code
first.** This is the jug-puzzle lesson applied:

- inventory as plain counts (it can read them; it just can't add them)
- **precomputed booleans:** `can_place_table`, `can_make_wood_pickaxe`,
  `have_enough_wood_for_goal`, `table_within_reach`, and so on
- nearby things given as **ranked lists** ("nearest tree", "second nearest"), not distances
- vitals and time of day as words: `drink: low`, `time: dusk`
- the current subgoal, stated in plain language

### Opus layer — the planner

- Called **once per episode** to turn "make a stone pickaxe" into an ordered
  subgoal list and the question wording for Jev.
- Called **again only on escalation** (see ⚖️) or after a subgoal fails twice.
- Never in the per-step loop.

Being honest about it: Crafter's tech tree is small enough that a hand-written
subgoal list works, so **Opus isn't strictly needed for the stone pickaxe.** It
earns its place on the iron run (survival trade-offs, getting through a night)
and in the escalation mechanism. That's why it comes in phase 3, not phase 1.

### ⚖️ Confidence-gated escalation — the demo doubles as an experiment

This is open question 3 from `FINDINGS.md`. When Jev's top choice comes back
below a confidence threshold, the loop pauses and asks Opus instead. The side
panel shows it happening ("Jev unsure, 0.41 → asking Opus"). The same run then
measures how often escalation fires and whether Opus's calls turn out better.

---

## 🎥 Video

- **Frame layout:** game on the left, rendered at ~600×600 with nearest-neighbor
  scaling so the pixel art stays crisp. Panel on the right: current subgoal, the
  Jev question, each option with a probability bar, confidence, latency,
  inventory icons, achievements unlocked, and any "asking Opus" banner.
- **Written straight to MP4** with imageio and ffmpeg (ffmpeg is already at
  `/opt/homebrew/bin/ffmpeg`), at a fixed ~8 fps. API latency never shows up in
  the file.
- **Decision frames are held about 1 second** so a viewer can read the panel.
  Ordinary walking frames play at normal speed.
- **Title and end cards** with the goal, then steps, Jev calls, Opus calls, cost
  and wall time. The end card also carries the **"best of N; K/N succeeded"**
  disclaimer and one **baseline line** (random / script / Jev), both generated
  from logged runs.
- **Night (iron run only):** brightness floor applied at render time, plus a
  "🌙 NIGHT" panel label. The game logic still sees true darkness.
- **`--live`** pygame window as a debug flag, documented in `run.py --help`. It
  draws the same composed frame as the MP4 writer.
- **Every decision is logged to JSONL** (state, question, answer, latency), so
  any moment in the video can be traced back to the call that caused it.

---

## 📏 Baselines — the jevsearch methodology, carried over

| agent | what it is | puzzle analogue |
|---|---|---|
| random | Crafter's own random policy over the 17 primitives | random |
| scripted | same skills, fixed hand-coded subgoal order, no model | oracle |
| Jev | same skills, Jev picks | Jev heuristic |
| Jev + Opus | Jev picks, Opus handles escalations | — |

Metrics over N seeds: success rate for stone and iron pickaxe, steps taken, calls
made, cost, deaths.

**A concern to state up front:** with a good skill layer, the scripted agent will
probably match or beat Jev on the stone pickaxe, because it's a short fixed
recipe. The demo isn't a claim that Jev plays better than a script. It shows a
general-purpose model **making the decisions visibly, in real time, from a
description of the world**, with no game-specific training. Jev's real
opportunity is the survival trade-offs on the iron run, where a fixed script is
brittle. The baselines keep us honest about which of the two we're showing.

---

## 🗂️ Directory layout

```
jevcraft/
  PLAN.md          this file
  pyproject.toml   deps; uv.lock pins every package; .python-version = 3.12
  .venv/           created by uv, not committed
  state.py         env → structured state: inventory, known map, booleans, rankings
  skills.py        macro-actions → primitive actions; BFS pathing (lava-safe)
  agents.py        RandomAgent, ScriptedAgent, JevAgent, JevOpusAgent
  chooser.py       builds Jev questions from state, calls Jev, parses answers
  planner.py       Opus: subgoals + question wording + escalation   (phase 3)
  render.py        frame + side panel composition, MP4 writer
  seeds.py         offline scan for seeds with reachable trees / stone / iron
  run.py           CLI: --agent --goal --seed --episodes --video --live
  results/         per-episode JSONL decision logs + summary stats
  videos/          MP4 output
```

> **Note for this copy:** the client *is* copied in, as `client.py`, so the
> project stands alone. The paragraph below describes the original working
> layout.

The Jev client is **imported from `../jevsearch/client.py`**, not copied. That
file already handles both backends and records the served model string.

---

## 🪜 Phases

Each phase ends with something you can run and look at. No phase before 2 makes
any API calls.

### Phase 0 — setup and smoke test *(no API)* ✅ built
- uv project: `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`
- `run.py`: random agent, MP4 via `imageio-ffmpeg` (H.264, yuv420p, so
  QuickTime plays it), `--live` pygame debug window, `-h` works even outside
  the environment
- checked in a scratch copy: 60-step run, 576×576 H.264 at 8 fps, and a frame
  looked at directly
- **done when:** you can open a Crafter video in QuickTime

### Phase 1 — skills, scripted agent, video pipeline *(no API)* ✅ built
- **Legibility comes first.** Ralph's reaction to the phase-0 video was "not clear
  what is going on." The camera follows the player, so without context the viewer
  is lost. The renderer adds:
  - **a minimap**: the known map at small scale, with the player dot, a fading
    **path trail**, and the current target marked
  - **a target highlight** in the main view: an outline on the tile being
    walked to (the tree, the stone)
  - **intent in words** at the top of the panel: "heading to nearest tree → need 4 wood, have 1"
  - **a bigger frame** (864px) and **slower playback** (~5–6 fps), with decision frames held
- `state.py`, `skills.py`, `render.py`, `seeds.py`
- the scripted agent reaches the stone pickaxe on a scanned seed
- side panel works, filled with the scripted agent's choices in the Jev slots
- **done when:** a scripted-agent video with the side panel reaches the stone pickaxe

This is where most of the effort goes, and none of it is Jev. Pathfinding, facing,
tunnelling and getting interrupted are the real engineering.

### Phase 2 — Jev chooser ✅ steps 1–2 built; step 3 (video) ready to record
- `chooser.py` plus `JevAgent`
- start with a `--dry` fake client, as in the jevsearch tests, to check the question shapes for free
- then run live on OpenRouter; spend is capped per episode, same as `run_exp.sh`
- **done when:** Jev reaches the stone pickaxe on the demo seed, and the logs show sensible choices
- **🎬 checkpoint video:** record the stone-pickaxe demo right away, with
  several takes, the best-of-N end card, and a baseline line. There's a shareable
  video before any iron work starts

### Phase 3 — Opus planner and escalation ✅ built, measured, recorded
- `planner.py`: subgoal list, question wording, escalation handler
- confidence threshold is tunable; start around the ~0.55 level where
  `min_ops` sat in Experiment A
- **done when:** a run shows at least one escalation on screen, and the logs record what Opus decided

### Phase 4 — evaluation *(the jevsearch habit)*
- 4 agents × N seeds, stone pickaxe first, then iron
- results in a short `FINDINGS.md`, with predictions written down beforehand, as before
- **done when:** we can say plainly whether Jev beat random, how it compares to
  the scripted agent, and whether escalation helped

### Phase 5 — record the demo
- pick a seed, run until we get a clean take, render with title and end cards
- **done when:** there's an MP4 you're willing to show people

---

## ⚠️ Risks

| risk | mitigation |
|---|---|
| Skill bugs (stuck on corners, facing the wrong way) take the most time | phase 1 is scripted-only, so model behavior can't hide pathing bugs |
| Agent dies to a zombie or lava | lava is never pathable; `in_danger` noul plus a flee skill; stone-pickaxe runs finish before dusk |
| No reachable iron on the chosen seed | `seeds.py` scans the full map offline before we commit to a seed |
| Jev isn't deterministic, so the same seed can play out differently | log everything; several takes, best-of-N disclosed on the end card (decision 6) |
| Night frames too dark to read | finish the stone run by step ~150; for the iron run, brighten the frames brightness floor plus night label (decision 7) |
| The scripted agent makes Jev look pointless | frame the demo as described in 📏; the iron run is where Jev has real work to do |
| Opus API key not available | Opus comes in phase 3; phases 0–2 don't need it |

---

## 💰 Cost

- Jev: ~20–60 calls per episode at ~$0.000036 each, so under a cent per episode
  and a few cents for the phase-4 evaluation.
- Opus: 1–5 calls per episode, plus escalations. Still small, but the main cost
  line, so it gets a cap per episode.

---

## ✅ Decisions (2026-09-19)

| # | question | decision |
|---|---|---|
| 1 | goal | **Stone pickaxe first, recorded as soon as phase 2 works**, then iron as an upgrade, not a requirement |
| 2 | map knowledge | **Explored-only** in the demo. Phase 1 builds against a known-map object filled with the full map, and the switch is a one-line change |
| 3 | Opus access | **Direct Anthropic API**, `ANTHROPIC_API_KEY` (already in env). Installed SDK is 0.49.0, so upgrade it inside the venv |
| 4 | Python env | **uv**: `pyproject.toml`, `uv.lock` (all 28 packages pinned with hashes) and `.python-version` (3.12). uv builds `jevcraft/.venv` from scratch and can fetch Python itself, so it runs for others exactly as it runs here. Pip users can get a requirements file with `uv export` |
| 5 | video | **MP4 is the deliverable**; `--live` pygame window is a debug flag. Every script supports `-h`/`--help`, and `--live` is documented there |
| 6 | takes | **Several takes, best one used.** The end card says "best of N runs on this seed; K/N succeeded", generated from logged counts, never typed by hand |
| 7 | night (iron run) | **Dim but readable**: brightness floor at render time only, plus a "🌙 NIGHT — zombies spawning" panel label |
| 8 | directory name | **`jevcraft`** |
| 9 | baselines in video | **Jev only on screen.** Baselines go on one end-card line (random / script / Jev success counts). One video to share. A separate side-by-side video stays possible later, most likely for the iron run if Jev beats the script there |

## ❓ Unresolved questions

None blocking. To revisit with real data:
- escalation confidence threshold (start ~0.55, tune in phase 3)
- night brightness floor (tune by eye on real frames)
- whether iron run earns a side-by-side video
