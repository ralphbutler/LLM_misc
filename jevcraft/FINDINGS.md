# 🎮 jevcraft — findings

**Who:** Ralph Butler
**When:** 2026-09-19
**Models:** Jev `jev-1.13.0` (TypeSafe API; OpenRouter behaves the same) and
Claude Opus 5 `claude-opus-5` (Anthropic API)
**Game:** Crafter 1.8.3, a 2D Minecraft-like benchmark
**Companion:** an earlier project, *jevsearch*, put Jev inside a search loop on
water-jug and blocks-world puzzles; it is not included here, and the findings it
contributed are quoted where they matter.
The full step-by-step log, including every mistake, is in `STATUS.md`.

**This document has two parts.** Part 1 is the **stone pickaxe**: a short task,
where a hand-written script is hard to beat. Part 2 is the **iron pickaxe**: a
long one, with digging, hunger and night — and the part where the models pull
ahead of the script. If you read one thing, read Part 2's results.

---

# Part 1 — 🪨 The stone pickaxe

## 🎯 The question

Jev is a "System One" model: fast, cheap, and it returns a calibrated choice,
not text. It cannot plan. In `jevsearch` we wrapped it in a search loop and it
played puzzles well, with one clear weakness: arithmetic.

Here we asked whether the same idea runs a *game*, in real time, from a
description of the world, and what happens when a slow "System Two" model
(Opus) sits behind it.

The goal: go from bare hands to a **stone pickaxe**. That means chopping wood,
placing a crafting table, crafting a wood pickaxe, mining stone, and crafting
the stone pickaxe, while zombies wander about. Four wood and one stone in all.

---

## 🏗️ How it works

```
Opus    slow, ~5 s, ~$0.01 a call   plans once; takes over only when Jev is unsure
  ▲ hand-off when Jev's confidence < 0.55
Jev     fast, ~0.2 s, ~$0.00004     picks the next step from a menu, every time
  ▲ menu of options that are possible right now
Skills  our code                    walking, chopping, placing, crafting, fleeing
Crafter the game
```

- **Our code owns the world.** A skills layer turns one choice ("collect
  wood") into keystrokes: pathfinding, facing, chopping, replanning around
  cows. It only offers options that are actually possible now.
- **Jev picks from that menu.** One call per decision, with the whole
  situation written as words and yes/no facts. It is asked only when a skill
  finishes or something interrupts, about 10 calls a game, not 400.
- **Opus is consulted rarely.** Once for a short plan, and again only when
  Jev's top choice comes back under 55% confidence.
- **The agent knows only what it has seen.** Unexplored land is unknown, and
  "explore" is one of the options.

Two baselines keep it honest: **random** key presses (the floor) and a
**hand-written script** that follows the recipe (the "perfect play" reference).

---

## 📊 Results

### Across 11 worlds, explored map, one run each

| who decides | reached the goal | mean steps | median steps | cost per game |
|---|---|---|---|---|
| hand-written script | 11/11 | 36.9 | — | free |
| **Jev alone** | 11/11 | 53.0 | 44 | ~$0.0003 |
| Jev + Opus plan | 11/11 | 38.8 | 35 | +$0.006 |
| **Jev + Opus hand-off** | 11/11 | 40.5 | **29** | +$0.011 |
| Jev + both | 11/11 | 43.4 | 31 | +$0.022 |

Leaving out one outlier world (see below), hand-off only averages **30 steps**,
the same as the script (~32).

### On the demo world (seed 21), several takes each

| who decides | result |
|---|---|
| random | 0/5 — never gets anywhere near a pickaxe |
| hand-written script | 5/5, median 49–54 steps |
| Jev alone | 102 steps (15 identical takes; later runs 84 and 104) |
| **Jev + Opus hand-off** | **5/5, all 38 steps** |
| Jev + Opus plan and hand-off | 5/5, 44–52 steps, full health in the best take |

**On this world, one cent of Opus, spent at the single moment Jev was unsure,
made Jev faster than the hand-written script.**

---

## 🔍 What we learned

### 1. Jev can run a game, if the code gives it facts, not sums
Jev reached the goal on every world we tried, agreeing with the script on 85%
of choices. What made the difference was **doing every sum in code first.**
One early menu said "collect wood (goal needs 4)" after 3 wood had already been
spent, which is exactly the subtraction Jev can't do. Rewritten as "have 1,
enough for the goal", the problem went away. This is the jevsearch
arithmetic finding, confirmed in a new setting.

### 2. The menu wording is part of the design
When a zombie approached, Jev kept choosing "back away" where the script
fights. It cost steps and saved no health, because **a zombie follows you** and
the menu never said so. Once the options stated their consequences ("it follows
and keeps pace", "usually costs 2–4 health") and their purpose ("both pickaxes
are crafted at a table"), Jev's choices improved. That isn't cheating: the code
knows these facts, and Jev's job is to judge, not to guess the rules.

### 3. Low confidence marks the bad decisions, so it is a hand-off signal
Jev's worst choices, such as exploring after it had already found everything
it needed, came with low confidence (15–38%). Handing exactly those moments to
Opus fixed them. Opus was consulted on about 1 decision in 7 and changed Jev's
choice about half the time. **The fast model makes nearly every decision; the
slow one only fills in where the fast one is unsure.**

### 4. Opus's two roles overlap rather than add up
A plan up front and a hand-off when unsure each captured most of the gain on
their own. Using both was no faster and cost twice as much. The plan also makes
runs more varied, because Opus writes a fresh plan every game.

### 5. A sensible decision can still cost a lot
On one world a zombie was adjacent and Jev was unsure. Opus chose to back away
("break contact first before wandering off to hunt for trees"), which is
reasonable, but it moved the player off the route to the trees, and finding
wood again took about 90 steps. Jev alone had simply kept exploring. Good
reasoning at the moment of choice doesn't guarantee a good outcome in a world
that keeps going.

### 6. Runs don't repeat exactly, and there are three reasons
- **Jev** isn't deterministic (jevsearch). Here its choices were very stable:
  15 takes in a row made the identical 10 decisions, but then it varied.
- **Crafter** stores creatures in Python sets whose order depends on memory
  addresses, so zombies behave a little differently each run, even with the
  same seed.
- **Crafter's night effect draws from the game's own random generator**, so
  simply drawing an extra frame at night changes what the zombies do. Our
  video code uses only the frames the game already produced.

That is why every video's end card says it is "best of 5" and gives the
baselines, all generated from the logs.

---

## 🎥 The videos

A 1440×864 MP4. The game is on the left with a yellow outline on the current
target; a panel on the right shows what the agent is doing, the full menu with
Jev's probabilities, and why the choice was made. Colored banners flag
events: red for danger, amber for "Jev unsure", blue for "asking Opus" and
"Opus overrode Jev", green for discoveries. A minimap shows the explored world.

- `videos/final_jev_seed21.mp4` — Jev alone
- `videos/final_jev-opus-noplan_seed21.mp4` — Jev + Opus hand-off (the clearest)
- `videos/final_jev-opus_seed21.mp4` — Jev + Opus plan and hand-off

---

## ⚠️ Caveats

- One run per world in the 11-world comparisons. The differences are clear but
  not statistically settled.
- One goal (stone pickaxe) and short games (20–150 steps), all in daylight
  except by accident. The iron pickaxe, with night and survival, is untested.
- The menu wording was improved after seeing Jev's behavior. Results before and
  after are not compared as if nothing changed; STATUS.md records exactly when.
- The skills layer does a lot of work (pathfinding, facing, digging). Jev
  chooses *what* to do, never *how*.

---

## 🧾 Honest record of our own mistakes

All of them were in the harness or the process, not the models:

- a variable-capture bug that would have sent "collect wood" to stone
- the "goal needs 4" arithmetic trap described above
- a menu that kept offering stone the agent could see but not reach (542
  failed attempts in one game), fixed by requiring a known path and resting any
  option that fails
- deleting run logs by wildcard, twice, including some of Ralph's
- a dry test run that wrote fake data into a real run's baseline, which
  reached an end card before it was caught and re-run

---

## ➡️ What Part 1 left open

The stone run ended with four questions. The first was the big one: **does the
Jev + Opus split still hold when the game is long and dangerous?** That is what
Part 2 answers.

---

# Part 2 — ⛏️ The iron pickaxe

## 🎯 Why a second, longer goal

On the stone run the hand-written script was hard to beat, and for a fair
reason: the recipe is five steps long, fixed, and nothing much happens while
you follow it. That is the ideal case for a rigid script and the worst case for
judgment.

The iron pickaxe is the opposite. The bill is **5 wood, 5 stone, 1 coal, 1
iron**, in a strict order — wood, table, wood pickaxe, stone, stone pickaxe,
then coal and iron, which can only be *dug out of the inside of mountains*. The
pickaxe must be crafted with a table **and** a furnace both within one tile. The
games run 3–10 times longer, so thirst, hunger and **night** all arrive before
the goal does. Zombies spawn in the dark. This is where a fixed recipe should
break.

Everything else is unchanged from Part 1: the same skills layer, the same menu,
the same Jev, the same 0.55 hand-off threshold, the same explored-only map.

---

## 📊 Results

### The headline: 11 worlds, two full passes, 22 runs per setup

| who decides | reached the iron pickaxe | median steps | Opus cost per game |
|---|---|---|---|
| hand-written script | 16/22 | 100 | free |
| Jev alone | 13/22 | 101 | ~$0.0006 |
| Opus plan only, no hand-off | **10/22** | 106 | ~$0.009 |
| **Jev + Opus hand-off** | **19/22** | **96** | ~$0.067 |
| **Jev + Opus plan and hand-off** | **22/22** | 106 | ~$0.10 |

```
per-world, both passes      21    1    2    3    4    5    6    7    8    9   10
plan only                   XX   oo   Xo   oo   oX   XX   oo   Xo   XX   XX   Xo
hand-off only               oo   oo   oo   oo   oX   oo   oo   oo   oo   XX   oo
plan + hand-off             oo   oo   oo   oo   oo   oo   oo   oo   oo   oo   oo
```

**Plan + hand-off reached the goal in all 22 runs**, on the same worlds where
the script failed 6 times and Jev alone failed 9. Seeds 5, 8 and 9 failed in
*every* scripted and Jev-alone run; the hand-off cleared 5 and 8, and adding the
plan cleared 9 as well.

This is the result the iron run was built to find. **On a short task the script
wins; on a long one it does not.**

### On the demo world (seed 21), five takes each

| who decides | result |
|---|---|
| random | 0/5 |
| hand-written script | 5/5, median 100 steps |
| Jev alone | 3/3, median 109 steps |
| **Jev + Opus hand-off** | **5/5 — 76, 77, 80, 76, 76 steps** |

Every take beat the script, at about 8 cents of Opus each. On the stone run the
same comparison came down to a single decision; here the margin is routine.

---

## 🔍 What we learned

### 1. The hand-off is the trick, and it is worth more the longer the game
Opus is consulted on roughly one decision in three on iron (against one in
seven on stone), because the bigger menu produces more low-confidence moments.
It **overrode Jev on 60%** of the decisions it was asked about. That is the
whole mechanism: Jev makes nearly every decision at $0.00004 and 0.2 s, and the
expensive model is spent only where the cheap one admits it is unsure.

### 2. A plan alone is the worst setup we tested — and the same plan plus the hand-off is the only perfect one
Opus writing a short plan before step 0, with no hand-off, scored **10/22** —
no better than Jev with no plan at all (13/22), and its failures have the same
shape every time: starvation or a daytime zombie around step 300, the agent
still following a recipe written before it had seen the world. Yet **that same
plan turns 19/22 into 22/22 when the hand-off is there too.**

The reading we believe: a plan is a good thing to have and a bad thing to obey.
It helps when something is watching for the moment it goes stale. On the stone
run the two roles looked redundant (Part 1, finding 4); on the long run they
are complementary, and the plan's value only appears in company.

### 3. Disagreement is where the value is
Jev agreed with the script on **85%** of stone decisions but only about **50%**
of iron ones. The iron menu is far bigger — mine coal, mine iron, place the
furnace, walk back to the table, build a second table, craft a sword, shelter —
so there is much more to be wrong about. The setups that win are the ones that
put a second opinion exactly where the disagreement is.

### 4. Fixing a survival failure moves it rather than removing it
Night is what kills the scripted agent. We built a shelter skill (dig a
two-tile room into rock, step back in so you face the entrance, wall it off,
sleep) and then found the agent rarely used it. A 160-run A/B of the fix:

| | before | after |
|---|---|---|
| reached iron | 49/80 | 50/80 |
| deaths at night | **20** | **9** |
| starvation | 5 | 11 |

Night deaths more than halved and **the success rate did not move**: the runs
that survived the night went on to starve the next day. Judge a survival fix by
the table of *causes*, not the total.

### 5. The bug was in our baseline, not in the model
Before blaming the models for dying at night, we checked whether "shelter" was
even on the menu when they died. It was — on one world it was offered **nine
times** while the scripted agent, carrying nine stone, spent the whole night
hunting a cow that did not exist. The shelter rule sat below the hunger and
fight rules in the script's priority order, so anything at all outranked it.
**The baseline you compare against is code you wrote, and it has bugs too.**

### 6. Measurement noise was the hardest part of this run
In a 40-world A/B of one change, 8 worlds flipped outcome and **no world failed
in both arms**: with one run per world, which world fails is close to a coin
toss. Several comparisons were over-read before we measured that. It is why the
headline table above is two full passes and 22 runs per setup, and why we still
say a 3-run margin (22/22 against 19/22) is a lean and not a finding.

---

## 🎥 The video

`videos/final_jev-opus-noplan_iron_seed21.mp4` — 39 s, the best of 5 takes on
seed 21, Jev + Opus hand-off, with the same panel as the stone videos plus a
night-brightness floor so the dark is watchable. The end card is generated from
the logs: the baselines, the takes, the calls and the cost.

Hand-off only is the recorded setup on purpose: on this world it is 76 steps
against plan + hand-off's 112–336, and it costs a third less.

---

## ⚠️ Caveats

- 11 worlds, two runs each. Enough to separate 22/22 from 16/22; **not** enough
  to rank 22/22 against 19/22.
- One demo world (seed 21) was chosen for being convenient to film, and the
  video's five takes are on that world only.
- The script we compare against is our own, and it is a real recipe, not a
  strawman — but a better script is always possible. Its honest record is
  "reaches iron on most full maps and about 60% of explored ones, and dies at
  night when it doesn't".
- The skills layer still does all the *how*: pathfinding, digging, facing,
  crafting. Both models only choose *what*.
- Opus cost per game grew by a factor of ten against the stone run (1 call to
  7–20), because the escalation rate rose with the menu size.

---

## 🧾 Honest record of our own mistakes (Part 2)

- A sloppy string replace pasted a constant into three unrelated lines and
  `run.py` stopped parsing; the error went to a background log, not to us, and
  a whole measurement silently never ran. We now parse-check after every edit.
- `takes.py` was hard-wired to the stone pickaxe and had no `--goal`, although
  the iron plan's own command assumed it.
- A dry rehearsal shared MP4 filenames with real takes and would have
  overwritten them — the second time a dry run nearly contaminated real data.
  Dry runs now carry their own name throughout.
- Several I-1 comparisons were over-read before the noise was measured; they
  are left in `STATUS.md` with that correction rather than deleted.

---

## ❓ Open questions

1. **Is plan + hand-off really better than hand-off alone?** 22/22 against
   19/22 is 3 runs; it needs more worlds, not more passes.
2. **Where should the hand-off threshold sit?** 0.55 was a first guess and has
   never been varied. On iron it fires on a third of decisions.
3. **What does Opus actually fix?** We have the overrides logged but have not
   classified them: how many are survival saves, how many are routing?
4. **Would better menu wording close the gap without Opus?** It did most of the
   work on stone, and Jev alone is still 13/22 on iron.
5. **A third goal** — diamond needs an iron pickaxe first, so the same question
   asks itself again one level up.
