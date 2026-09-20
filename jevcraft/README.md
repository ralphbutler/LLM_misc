# 🎮 jevcraft — a fast model plays Crafter, a slow one helps when it's stuck

Two models play a 2D Minecraft-like game together, and the code keeps score.

- **Jev** (TypeSafe) is a *System One* model: ~0.2 s, ~$0.00004 a call. It does
  not write text — it returns a **typed, calibrated choice**. It picks the next
  thing to do, every time.
- **Claude Opus 5** is the *System Two* half: ~5 s, ~$0.01 a call. It is asked
  only when **Jev's confidence drops below 0.55**, and it chooses from the same
  menu Jev saw.
- **The code owns the game.** It offers a menu of what is possible right now,
  does every sum, and turns one choice ("mine the iron") into keystrokes.

Everything is recorded to an MP4 with a side panel showing each decision, the
menu, Jev's probabilities, and when Opus took over.

**The result, on the long goal (iron pickaxe, 11 worlds, 22 runs per setup):**

| who decides | reached the goal |
|---|---|
| random key presses | 0 |
| hand-written script | 16/22 |
| Jev alone | 13/22 |
| Opus plans, no hand-off | 10/22 |
| **Jev + Opus hand-off** | **19/22** |
| **Jev + Opus plan and hand-off** | **22/22** |

The write-up is [`FINDINGS.md`](FINDINGS.md). The short version: on a *short*
task a hand-written script is hard to beat; on a *long* one, asking a slow model
for help at the moments the fast model is unsure beats everything, and it costs
about 7 cents a game. A plan by itself, with no hand-off, is the **worst** setup
here — it goes stale — yet the same plan plus the hand-off is the only one that
never failed.

Everything here is reproducible, including the videos:
[**reproduce the results**](#-reproduce-the-results) gives the exact command and
cost for each table, and [**the videos**](#-the-videos-and-how-to-make-them)
gives the one-line command behind each recording.

---

> **Where this lives.** `jevcraft` is one project inside the `LLM_misc`
> repository, self-contained in this folder: its own dependencies, its own
> `.gitignore`, its own videos. **Run every command below from this directory**
> (`cd LLM_misc/jevcraft`) — that is how `uv` finds the project.

## 🚀 Quick start

### 1. Set up the environment (once)

```bash
uv sync
```

That is the first thing to do. It reads `pyproject.toml` and `uv.lock`, fetches
**Python 3.12** if you don't have it, and builds a **`.venv/` of about 135 MB**
in this directory — the game, numpy, the video encoder and the two HTTP clients.
It takes a minute or two on a first run and is instant afterwards. `.venv/` is
git-ignored: it is yours, nothing commits it, and it is rebuilt by running
`uv sync` again.

If you skip this step, the first `uv run` below does it for you — the pause is
the download, not the game.

### 2. Play a game, free, with no API keys

```bash
uv run python run.py --agent scripted --seed 21          # a couple of seconds
open videos/scripted_seed21.mp4
```

That runs the hand-written baseline and writes a video. Every `uv run python`
command uses the environment from step 1; there is never anything to activate.

### 3. Let the models play

With keys (below):

```bash
uv run python run.py --agent jev --seed 21 --map explored             # Jev alone, ~$0.0005
uv run python run.py --agent jev-opus --seed 21 --map explored        # Jev + Opus
uv run python run.py --goal iron_pickaxe --agent jev-opus --seed 21 --map explored --no-opus-plan
```

**No keys? Everything still runs.** `--dry` replaces both models with fakes that
answer at random. It is free, it exercises the whole pipeline, and it is the
right way to rehearse a recording:

```bash
uv run python run.py --agent jev-opus --seed 21 --dry
```

### Requirements

- [`uv`](https://docs.astral.sh/uv/) is the only thing to install yourself
  (`brew install uv`, or see their site). It handles the rest, including
  **Python 3.12**, which `.python-version` pins.
- **ffmpeg** arrives with the `imageio-ffmpeg` dependency — no system install.
- About **135 MB** of virtual environment. Nothing else is downloaded.
- Developed on macOS (Apple Silicon); nothing in it is platform-specific.

### Keys

| variable | what it is for | needed by |
|---|---|---|
| `TYPESAFE_API_KEY` | Jev, via TypeSafe's own API (the default) | `--agent jev`, `--agent jev-opus` |
| `OPENROUTER_API_KEY` | Jev, via OpenRouter (`--backend openrouter`) | the same, with `--backend openrouter` |
| `ANTHROPIC_API_KEY` | Claude Opus 5 | `--agent jev-opus` only |

**Both backends serve the same model and we could not tell them apart** — same
choices, same confidences, within the model's own run-to-run variation. Use
whichever you have. `TYPESAFE_BASE_URL` overrides the API host if you need it.

Every program names the missing key *before* doing any work, and `-h/--help`
works without keys.

---

## 🔗 How Jev and Opus fit together

```
Opus    slow, ~5 s, ~$0.01/call    plans once; decides only when Jev is unsure
  ▲ hand-off when Jev's confidence < 0.55   (--escalate)
Jev     fast, ~0.2 s, ~$0.00004    picks the next option from the menu, every time
  ▲ menu of options that are possible right now
Skills  this code                  walk, chop, mine, dig, place, craft, fight, flee, shelter
Crafter the game
```

**The loop.** The agent is consulted *between skills*, not every frame — about
10–25 times a game rather than 400. At each decision point the code builds the
menu of options that are genuinely possible now, writes the situation out in
words and yes/no facts, and asks Jev once. Jev returns a probability over the
options; the top one is taken.

**The hand-off.** If Jev's top choice comes back under `--escalate` (0.55 by
default), that single decision goes to Opus, which sees the same menu and the
same situation and answers with a choice plus a one-line reason for the video.
Opus is consulted on roughly 1 decision in 7 on the short goal and 1 in 3 on the
long one, and it overrides Jev about 60% of the time it is asked.

**The plan.** `--agent jev-opus` also has Opus write a short ordered plan before
step 0, shown to Jev on every call afterwards. `--no-opus-plan` keeps the
hand-off and drops the plan. See the headline table for why that matters.

**Two rules that shaped the whole design:**

1. **Every sum happens in code.** Jev judges structure well and arithmetic
   badly, so menus say *"have 2, 3 more needed"* and never *"the goal needs 5"*.
2. **Options state their consequences.** "it follows and keeps pace, so it will
   be back", "usually costs 2–4 health". The code knows these facts; Jev's job
   is to judge, not to guess the rules.

---

## 🕹️ The programs

Each takes `-h/--help`, which lists every option with its default and worked
examples. All of them write to `results/` (logs) and `videos/` (MP4s).

### `run.py` — one episode, one video

```bash
uv run python run.py --agent jev-opus --seed 21 --map explored
uv run python run.py --goal iron_pickaxe --seed 21 --map explored     # the long run
uv run python run.py --agent jev --seed 21 --backend openrouter
```

| option | what it does |
|---|---|
| `--agent {random,scripted,jev,jev-opus}` | who decides. `random` is the floor, `scripted` the hand-written reference |
| `--goal {stone_pickaxe,iron_pickaxe}` | short goal or long one (default: stone) |
| `--seed N` | the world. The **map** is fixed by the seed; creatures are not |
| `--map {full,explored}` | whether the agent knows the whole map or only what it has seen |
| `--escalate 0.55` | hand a decision to Opus below this confidence. `0` disables |
| `--no-opus-plan` | Opus hand-off without the up-front plan |
| `--backend {typesafe,openrouter}` · `--model` | where Jev is called |
| `--max-calls` · `--opus-max-calls` | hard spend guards per episode |
| `--dry` | fake models, free, no keys |
| `--no-video` · `--fps` · `--hold` · `--out` · `--tag` | recording controls |
| `--live` | a pygame window while it plays (debugging) |

Outputs: `videos/<agent>_seed<N>.mp4`, plus `results/<agent>_seed<N>_<tag>.jsonl`
— one JSON line per decision, holding the whole menu, Jev's probabilities, the
choice, the confidence, whether Opus was asked and what it said, position,
inventory and daylight — and a `_summary.json` beside it.

### `compare.py` — the measurement harness

Runs N worlds × M setups × R runs in parallel and prints the comparison table.
**Every table in `FINDINGS.md` came from this.**

```bash
uv run python compare.py --goal iron_pickaxe --setups scripted --runs 2   # free
uv run python compare.py --setups jev,jev-opus --dry                      # free rehearsal
```

| option | what it does |
|---|---|
| `--setups` | comma-separated: `random`, `scripted`, `jev`, `jev-opus-noplan`, `jev-opus-plan`, `jev-opus` |
| `--seeds 21,1-10` | the worlds; ranges allowed |
| `--runs 1` | runs per world per setup — **use 2+ before believing a small gap** |
| `--goal` · `--map` · `--backend` | as `run.py` |
| `--jobs 4` | games in parallel |
| `--keep-logs` · `--out rows.json` | keep the per-decision logs / dump the raw rows |
| `--dry` | fake models, free |

It prints a per-setup summary, a per-world grid of o/X, every failure with its
cause, and the total cost.

### `takes.py` — the presentable video

Records N takes on one seed, runs the baselines, picks the best take (**reached
the goal → most health → fewest steps**) and appends an end card generated
**from the logs**, so the card cannot flatter the run.

```bash
uv run python takes.py --seed 21 --map explored --goal iron_pickaxe --no-opus-plan
uv run python takes.py --seed 21 --dry --takes 2          # free rehearsal
```

`--takes 5`, `--baselines 5`, `--jev-baseline 3`, `--pick N`, `--agent {jev-opus,jev}`,
plus `--seed/--map/--goal/--dry`. Outputs every take to `videos/takes/`, the
chosen one to `videos/final_<variant>_seed<N>.mp4`, and the record to
`results/takes_<variant>_seed<N>.json`.

### `seeds.py` — find a good world

Offline, no API, no game played: it reads the map a seed generates and ranks
seeds by distance to trees, stone, water, coal and iron, and how much lava is
close by.

```bash
uv run python seeds.py                 # score seeds 1-50 for the stone run
uv run python seeds.py --iron          # ...for the iron run instead
```

Seed 21 is the demo world in every video here, and it wins both rankings.

---

## 🔁 Reproduce the results

Times are for an 8-core laptop; costs are what we actually paid.

| table in `FINDINGS.md` | command | cost |
|---|---|---|
| the baseline script, iron, 40 worlds | `uv run python compare.py --goal iron_pickaxe --setups scripted --seeds 1-40 --runs 2 --jobs 8` | free, ~20 min |
| Jev alone vs the script (11 worlds) | `uv run python compare.py --goal iron_pickaxe --setups scripted,jev --runs 2 --jobs 4` | ~$0.02 |
| **the headline: three Opus setups** | `uv run python compare.py --goal iron_pickaxe --setups jev-opus-plan,jev-opus-noplan,jev-opus --runs 2 --jobs 4` | **~$4** |
| the stone-run comparison | `uv run python compare.py --goal stone_pickaxe --setups scripted,jev,jev-opus-noplan,jev-opus --runs 2` | ~$0.6 |
| the iron video | `uv run python takes.py --seed 21 --map explored --goal iron_pickaxe --no-opus-plan` | ~$0.40 |

**Expect your numbers to differ a little, and expect the *same shape*.** Which
individual worlds fail is close to a coin toss (see below); the ordering of the
setups is what reproduced across two full passes for us.

Start with `--dry` on any of them to check the pipeline for free.

---

## 🧩 The supporting modules

Not run directly — imported by the programs above.

| file | what it holds |
|---|---|
| `state.py` | what the agent is allowed to know. `KnownMap` is either the whole map or only seen tiles (one flag), plus pathfinding that **digs through rock** when the right pickaxe is held |
| `skills.py` | the skills — walk, collect, mine, place, craft, eat, drink, fight, flee, explore, shelter — and **`menu()`**, which decides what is possible right now and writes each option's label. This file defines the models' whole world-view |
| `agents.py` | the non-model agents: `RandomAgent` (the floor) and `ScriptedAgent`, the hand-written recipe, with its survival behaviours as switches (`swords`, `shelter`, `eat_early`, `shelter_first`) |
| `chooser.py` | the **Jev** agent: builds the situation, shuffles the options (order alone can flip a top choice), makes one call, returns the decision with its probabilities |
| `planner.py` | **Opus**: `plan()` once per episode, `decide()` only on a hand-off. Structured output constrains the answer to the menu's keys, so it cannot reply with something that isn't on offer |
| `render.py` | the 1440×864 frame: game view, yellow outline on the current target, side panel, minimap, alert banners, end cards |
| `client.py` | a small Jev client over `httpx`; both backends share one request shape |

---

## 📁 Output and documentation

```
results/   one .jsonl + one _summary.json per run — every decision, replayable
videos/    created on your first run: the finished MP4s, with videos/takes/
           holding every take and end card from a takes.py session
```

| document | what it is |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | **the write-up** — Part 1 the stone pickaxe, Part 2 the iron one |
| [`STATUS.md`](STATUS.md) | the lab notebook: every step dated, including the mistakes and the measurements that overturned earlier conclusions |
| [`PLAN.md`](PLAN.md) / [`PLAN_IRON.md`](PLAN_IRON.md) | the plans, and the decisions settled before the work started |

`STATUS.md` is long and unedited on purpose. If you want to know why something
is the way it is, it is in there.

---

## 🎥 The videos, and how to make them

**The MP4s are not in the repository** — they are a few megabytes of binary in a
repo of text, and every one of them is reproducible from a single command. What
follows is what each one shows and how to make it yourself.

Each is a **best-of-5 on seed 21**, the demo world, with the agent knowing only
what it has seen (`--map explored`). None is a lucky run picked by hand:
`takes.py` records five takes, chooses by a fixed rule — **reached the goal →
most health → fewest steps** — and appends an end card built from the run logs,
so the card states the baselines and the failures whether they flatter the run
or not. Your own takes land in `videos/` with their records in
`results/takes_*.json`.

**What you will see:** the game on the left with a yellow outline on the tile being
worked toward; on the right, what the agent is doing in words, the full menu
with Jev's probabilities, and why the choice was made. Banners flag events — red
for danger, amber for "Jev unsure", blue for "asking Opus" and "Opus overrode
Jev", green for discoveries. A minimap shows how much of the world has been
seen. The iron video adds a brightness floor at night so the dark is watchable
(the *game* still sees true night).

| the command | what it shows | what we got |
|---|---|---|
| `uv run python takes.py --seed 21 --map explored --agent jev` | **Jev alone**, stone pickaxe — no Opus at all | 102 steps, health 9, 10 Jev calls (~$0.0004). All five takes landed on 102–104 steps |
| `uv run python takes.py --seed 21 --map explored --no-opus-plan` | **Jev + Opus hand-off**, stone — the clearest demonstration | 38 steps, **1 Opus call** (~$0.01) at the one moment Jev wavered, and it changed the choice. All five takes were exactly 38 |
| `uv run python takes.py --seed 21 --map explored` | **Jev + Opus plan and hand-off**, stone | 44 steps, 2 Opus calls (a plan + 1 hand-off, ~$0.026). The plan made runs *more* varied: 44–52 steps |
| `uv run python takes.py --seed 21 --map explored --goal iron_pickaxe --no-opus-plan` | **the iron run** — Jev + Opus hand-off on the long goal | 80 steps, 14 Jev calls, **6 Opus hand-offs that changed 3 choices** (~$0.059). Takes ranged 76–80; that one won on health |

Each costs a few cents (the iron one about $0.40 including its baselines), and
each also runs `random` and `scripted` on the same seed for the end card's
comparison line. Add `--dry` to rehearse any of them for free.

**The first two are worth making and watching back to back:** same world, same
model, 102 steps against 38. The difference is one Opus call.

---

## 🎲 Reproducibility, honestly

- **The map is fixed by the seed; the game is not.** Crafter keeps creatures in
  Python sets whose iteration order depends on memory addresses, so zombies
  differ run to run on the same seed.
- **Never call `env.render()` yourself.** Crafter draws its night darkness from
  the world's own random generator, so an extra render changes what the zombies
  do. Use the frame `env.step()` returns — the video code does.
- **Jev is not deterministic either** (±0.05 on mid-range confidences), which is
  why it is asked once per decision and then commits.
- **Measurement noise is large.** In a 40-world A/B of one change, 8 worlds
  flipped and *no world failed in both arms*. Single runs per world are close to
  coin tosses; the headline table is two full passes, 22 runs per setup, and we
  still treat a 3-run margin as a lean rather than a finding.

---

## 💰 What it costs to run

| | per game |
|---|---|
| random / scripted / `--dry` | free |
| Jev alone (10–25 calls) | $0.0004 – $0.0009 |
| Jev + Opus, hand-off only | ~$0.011 stone · ~$0.067 iron |
| Jev + Opus, plan and hand-off | ~$0.022 stone · ~$0.10 iron |

A five-take recording with baselines is about $0.40 on the iron goal.
`--max-calls` and `--opus-max-calls` are hard per-episode guards.

---

## 🙏 Credits

[Crafter](https://github.com/danijar/crafter) by Danijar Hafner (MIT) is the
game and the benchmark; this project adds an agent, a menu, and a camera.
