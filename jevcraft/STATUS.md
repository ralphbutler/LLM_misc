# 📍 jevcraft — status

Running log of what we've done, newest state at the top. `PLAN.md` holds the
plan and its decisions; this file tracks progress against it.

> **Reading this outside the working directory:** this is the unedited lab
> notebook, so it refers to a few things that were deliberately not copied here
> — the emails sent to the model's authors (`EMAIL_*.txt`), a session handoff
> memo (`HANDOFF.md`), and the puzzle project `../jevsearch/`, part of the
> original Jev project, which we used to guide the development of jevcraft.
> Everything it says about the code and the measurements applies here as
> written.

---

## 🧭 Where we are

| | |
|---|---|
| **phase** | 5 (iron) — **I-0 … I-4 all done**, I-3 confirmed on a second pass. Iron video: `videos/final_jev-opus-noplan_iron_seed21.mp4` |
| **next** | a new email file (Ralph, tomorrow) — the write-up and README are done |
| **then** | possibly a git repo: `README.md` is written for that |
| **blockers** | none |
| **untested** | `--live` at 0.8 scale |

## ✅ Phase checklist

- [x] plan written, 9 decisions settled
- [x] **phase 0**: uv project + random agent + MP4 + `--live`
- [x] **phase 1**: skills, scripted agent, side panel (no API)
- [x] **phase 2 step 1**: Jev chooser (TypeSafe or OpenRouter, `--dry` fake), agreement measured
- [x] **phase 2 step 2**: real decisions — flee option, explored-only map + explore
- [x] **phase 2 step 3**: 🎬 **stone-pickaxe checkpoint video**: `videos/final_jev_seed21.mp4`
- [x] **phase 3**: Opus planner + confidence-gated escalation; 11-world comparison; two videos
- [x] **phase 4**: evaluation (random / script / Jev / Jev+Opus on 11 worlds) + `FINDINGS.md` + `EMAIL_phase4.txt`
- [x] **phase 5**: iron run — I-0 script, I-1 survival, I-2 Jev, I-3 Jev+Opus, I-4 video

---

## 📜 Log

### 2026-09-19 — background (before jevcraft)

- Explored the upstream `jev-explore` repo (read-only, not ours).
- Built `../jevsearch/`: a search loop with Jev as the heuristic, run on water-jug
  and blocks-world puzzles. Full results in `../jevsearch/FINDINGS.md`. The
  findings that shape jevcraft:
  - Jev is excellent at structural judgments and weak where arithmetic is
    needed → **compute anything numeric in code and hand Jev booleans.**
  - Jev isn't deterministic: mid-range values move ±0.05 → **call it on events,
    not every step, and commit to a choice once made.**
  - Averaging repeated calls doesn't help → **don't pay for precision.**
  - Low confidence marks the questions it can't answer → **escalate to Opus there.**
  - OpenRouter and direct TypeSafe are indistinguishable → either backend works.
- Sent the findings email (`../jevsearch/EMAIL1.txt`).

### 2026-09-19 — plan

- Read the Crafter 1.8.3 source directly (downloaded, not installed) to check
  its rules: resource costs, crafting needs the table within 1 tile, the full map
  is exposed, lava kills instantly, the day/night timing.
- Wrote `PLAN.md`: three layers (Opus plans, Jev picks, skills execute), the
  video design, baselines, phases, risks.
- Settled all 9 open questions one at a time:
  stone first, recorded early, then iron · explored-only map (full map during
  phase 1) · Opus via the direct Anthropic API · uv environment · MP4 deliverable
  plus `--live` debug flag · several takes, best one used, disclosed on the end
  card · night dim but readable, with a label · name stays `jevcraft` · Jev only
  on screen, baselines on the end card.

### 2026-09-19 — phase 0 built

- **Switched from pip/venv to uv** at Ralph's suggestion: `pyproject.toml` +
  `uv.lock` (28 packages, hashed) + `.python-version` (3.12). uv runs its own
  CPython 3.12.9, separate from the system Python.
- `run.py`: random agent over Crafter's 17 actions, H.264/yuv420p MP4 at 8 fps
  (576px frames, divisible by 9 for tiles and 16 for the codec), `--live` pygame
  window, `-h` with examples that works even outside the environment.
- Checked in a scratch copy: 60 steps produced a valid 576×576 H.264 file; a
  frame was inspected by eye and renders correctly.
- `.gitignore`: `.venv/`, `videos/`, `results/`, `__pycache__/`, `rmb_*`.
- Noted: Ralph's shell has `VIRTUAL_ENV` set to a base venv, so uv warns and
  uses `jevcraft/.venv` instead. The warning is harmless.

### 2026-09-19 — phase 0 run by Ralph ✅

- `uv run python run.py`: 191 steps, died; `open` played it in QuickTime.
  `--live` window worked. Phase 0 is done.
- **Ralph's reaction: "not clear what is going on."** Plays fast, the frame is
  small, the player stays centered, and cows come and go with no obvious intent.
  This is the most important design input so far. Causes:
  - **the camera follows the player**, so the world scrolls under him and there's
    no sense of where he is or where he's heading
  - **a random agent has no intent to show**: it changes direction almost every
    step, which reads as frantic
  - **cows and zombies are spawned and despawned by Crafter's own balancing**
    around the player, so they appear and vanish on their own
- **Death cause, from replaying the run:** zombie attacks at nightfall. Health
  9→7→5→0 at steps 168/174/180, daylight 0.26→0.14, a hostile adjacent every
  time, food/drink/energy all still above 0. The last hit (5→0) was bigger than
  the first two, consistent with Crafter's extra damage to a sleeping player,
  since the random agent presses `sleep`. This confirms the plan's night risk.
- **Found: Crafter runs are not exactly reproducible.** Same seed, same agent,
  separate processes: deaths at 179, 179, 179, 183 (and 191 in Ralph's run).
  Cause: `engine.World` keeps creatures in `defaultdict(set)`, and set order
  depends on memory addresses, so spawn/despawn choices diverge. The **map** is
  still identical per seed; creature dynamics are not. The `run.py --help` text
  had claimed full reproducibility, and it's now corrected.
- **Phase 1 now leads with legibility** (added to PLAN.md): bigger frame, a
  minimap with a path trail, the target tile highlighted, the current intent
  written in words, slower playback.

### 2026-09-19 — phase 1 built ✅

- **New modules:** `state.py` (known map, state snapshot, Dijkstra pathing,
  facts computed in code), `skills.py` (collect / go-to / place / make / attack /
  sleep, plus `menu()`), `agents.py` (random, scripted), `render.py` (panel,
  minimap, cards, MP4, live), `seeds.py` (offline seed ranking). `run.py` rewritten.
- **Engine facts checked in the source before building:**
  - chopping a tree turns it to grass, one wood per tree
  - zombie damage is 7 on a sleeping player and 2 awake, with a 5-step cooldown
  - you can only turn toward a tile by trying to walk into it, so placing a
    table needs a two-step approach
  - walking into lava always "succeeds"
  - **Crafter's night effect draws noise from the world's RNG**, so any extra
    `env.render()` changes zombie behavior. The renderer uses only the `obs`
    that `env.step()` returns.
- **Results, scripted agent:** 40/40 seeds reach the stone pickaxe, median 30
  steps, max 50, zero skill failures, all long before dusk (~150). Random
  agent: dies at ~181 with nothing useful.
- **Seed scan:** seed 21 is the best candidate for both runs: first tree in 3
  steps, 11 trees nearby, stone 6, iron 10, no lava within 12.
- **Legibility (Ralph's phase-0 complaint):** the panel shows the intent in
  words, the full menu with the chosen option and a one-line why, inventory
  icons, and achievements. The minimap has the player, a fading trail, the view
  box and the target. The target tile gets a yellow outline in the game view.
  864px game view, 6 fps, decisions held 1.2 s. Frames checked by eye.
- **Bugs caught while building:**
  - late-binding lambdas in `menu()` would have given the wood skill the
    stone target
  - the menu showed the TOTAL wood bill ("goal needs 4") after 3 wood had
    already been spent. That's an arithmetic trap for Jev. Now `remaining()`
    computes what is *still* needed, and labels read "have 1, enough for the goal".
- **Seen in logs:** daytime zombies do appear. Seed 16 ended with health 4
  after fighting bare-handed (a zombie takes 5 hits). **Fight-or-flee** is a
  real decision, and there is no flee option yet. That's a candidate for phase 2.
- Test outputs removed from `results/`. Ralph's `videos/random_seed1.mp4` kept.

### 2026-09-19 — phase 2, steps 1–2 ✅

- **`chooser.py`:** `JevAgent` with the same `decide(state, options)` interface
  as the script. One call per decision, three questions: `next_step` (a choice
  over the menu, shuffled per call), plus `in_danger` and `should_recover`
  (yes/no, shown as annotations). The situation Jev sees is all words and
  booleans: vitals as "fine / getting low / critical", distances as "close /
  short walk / far", resources as "enough for the goal" or "1 more needed".
  One-option menus skip the call. `--dry` uses a fake Jev with the same response
  shape, at no cost. **Default backend: TypeSafe direct**; `--backend openrouter`
  also works.
- **Step 2, giving Jev real decisions:** a `flee_zombie` option, the explored-only
  map (`--map explored`, creatures known only while on screen), and an `explore`
  skill that walks one heading until the missing material is *reachable*.
- **Attention cues (Ralph's request):** colored banners over the game view (red:
  zombie adjacent, −N health, could not…; amber: zombie approaching, "Jev
  unsure: N%"; green: found X). In the panel: danger in red when ≥50%, low
  confidence in amber, and a **magenta line whenever Jev differs from the script**.
  The explored minimap zooms to the seen area.
- **Bugs found and fixed:**
  - **Seed 27 looped 542 times:** stone was seen but unreachable, and "mine stone"
    kept being offered. Fixed on two levels: options now require a *known path*,
    not just a sighting, and any option that fails is rested for 10 steps, so no
    chooser (script or Jev) can loop on it.
  - A `RuntimeError` stand-in would have hidden real bugs as "api error".
  - Dry runs reported a cost; they now report $0.
- **What Jev did, first live runs (seed 21):**
  - Full map: backed away from zombies 5 times where the script fights. It
    ended at 107 steps and health 6, against the script's 50 steps and health 7:
    **the caution didn't pay**, because zombies follow. The flee label had never
    said so.
  - **Label change after seeing this (noted so before/after runs are not
    compared as if nothing changed):** flee now says "it follows and keeps pace,
    so it will be back"; fight says "usually costs 2-4 health"; place, make and
    go-to state their *purpose* ("both pickaxes are crafted at a table").
  - Explored map: explored before placing the table (93%), then later placed it
    next to the stone and needed no walk back. That looks clever, but it was
    **partly accidental**: it also chose "explore" 3 more times after everything
    was reachable (76%, 52%, 37%). **Its confidence was low on the bad picks
    (29%, 15%).** Those are exactly the moments phase 3's hand-off to Opus is for.
- **Evaluation, Jev on the explored map, 11 worlds (seeds 21, 1–10), after the
  label change:** **11/11 reached the goal**, 81 calls total (~$0.003), agreed
  with the script on 69/81 choices (85%).
  - Versus the script on the same worlds: Jev **53 steps / health 8.9** on
    average, script **37 steps / 8.5**. Slower, a little safer. The health
    difference comes from seeds 21 and 5, where the script fought. One Jev run
    per world, so this is suggestive, not established.
- **`takes.py`:** runs the baselines (random, scripted) on the seed, N Jev takes
  with video, and picks the best (success, then health, then fewest steps). It
  renders the end card from the logs ("best of N; K/N succeeded", the baseline
  line, calls, cost, agreement, served model) and joins it to the take with
  ffmpeg. A dry rehearsal produced a truthful "0 of 2" card.
- **Housekeeping slip:** clearing test outputs I ran `rm -rf results/*` without
  listing it first. That also removed the decision log and summary from Ralph's
  own `run.py --seed 21` run. His video was untouched, and the log comes back by
  re-running. `results/` now holds only the 11 Jev evaluation logs.

### 2026-09-19 — first real takes.py run: missing key

- Ralph's run crashed with a traceback **after** the 10 free baseline runs:
  `TYPESAFE_API_KEY` wasn't in his shell. The key was added to his key file
  after that terminal started, so `.zshrc` never loaded it. My session had it,
  which is why my tests passed.
- Fixed: `takes.py` checks the key **before** doing any work and prints three
  ways forward (set the key, `--backend openrouter`, or `--dry`). `run.py`
  gives the same clean message instead of a traceback.

### 2026-09-19 — 🎬 checkpoint video recorded ✅

- `takes.py --seed 21 --map explored`, TypeSafe backend, served `jev-1.13.0`:
  - baselines on seed 21: **random 0/5, script 5/5**
  - **Jev 5/5**: 102, 102, 102, 104, 104 steps, all at health 9, 10 calls each
    (~$0.0004 per take), 6/10 agreement with the script each time
  - chosen: take 1 (automatic). End card generated from the logs, as designed.
- **All 5 takes made exactly the same 10 decisions, in the same order:**
  collect wood → explore → flee → explore → explore → place table → make wood
  pickaxe → mine stone → go to table → make stone pickaxe. Confidence varied
  between takes, sometimes a lot (decision 5: 0.30–0.62; decision 10:
  0.63–0.97), but the top choice never flipped. Jev's non-determinism showed up
  in *how sure* it was, not in *what it chose*, on this world. It also means
  "best of 5" here is representative, not cherry-picked.
- **Ralph: the minimap was jerky.** The zoom-to-explored view rescaled every
  time the explored area grew. Replaced with a **fixed-scale fog of war**: the
  whole world, unseen land dark, filling in as it's explored, never rescaling.
  Checked on a render. Trade-off: the explored patch is small at this scale.
  Option offered: a fixed 40×40-tile window around the start, which is 1.6×
  larger and widens at most once.
- The recorded checkpoint video still has the old zooming minimap. A re-record
  costs about $0.002.

### 2026-09-19 — minimap: fixed 40×40 window ✅

- Ralph picked the 40×40 option. In explored mode the minimap now shows a
  **fixed 40×40-tile window centered on the starting point**. It never moves or
  rescales while the player stays inside, and it's 1.6× the whole-world scale.
  If the player comes within 2 tiles of its edge it **widens to the whole
  world once, permanently**, and never shrinks back. Full-map mode still shows
  the whole world.
- Checked: a render on seed 21 (explored patch, trail, table, stone and target
  all readable). A forced test of the edge case: window (12,12,40) → the player
  reaches the east edge → (0,0,64) → back in the middle, still (0,0,64). Four long
  explored runs (seeds 5, 7, 16, 27) rendered without errors.

### 2026-09-19 — re-recorded with the 40×40 minimap

- `takes.py --seed 21 --map explored` again: random 0/5, script 5/5, **Jev
  5/5, and this time all five takes were 102 steps**, health 9, 10 calls, 6/10
  agreement. With the earlier batch, that's **10/10 identical decision
  sequences** on this world.
- **Ralph saw the minimap jump back toward the center partway through.** It
  was the widen-once switch, not a cut between takes (the final video is one
  take). Jev's exploring took it far northwest: already at the window edge by
  step 35 at (12,23), then (6,3), and it finished at (31,2) near the top of
  the world. So the window widened a third of the way in, and the player's
  dot jumped because the whole-world view is centered on the start.
- Proposed: replace widen-once with a **sliding window** (game-camera style).
  It stays 40×40, keeps still while the player is well inside, and slides one
  tile at a time near an edge, so the scale never changes and nothing jumps.
  Awaiting Ralph's go-ahead.

### 2026-09-19 — minimap: sliding window ✅ (undo: `--minimap widen`)

- Added `--minimap slide|widen` to `run.py` and `takes.py`, **default slide**.
  Both behaviours stay in the code, so undoing is a flag, not a revert.
  - **slide:** the window is 40×40 tiles at a fixed scale. It stays still while
    the player is more than 6 tiles from its edge, then slides one tile per
    step to keep the player in view, stopping at the world's edge. The scale
    never changes, so nothing jumps.
  - **widen:** the previous behaviour. It switches to the whole world once.
- Checked:
  - Traced Jev's real seed-21 route shape: slide went (12,12,40) → (11,…) →
    (10,…) → (4,…) → (0,0,40) and stayed at 40; widen jumped to (0,0,64).
  - A live Jev run on seed 21 (reached the goal, ~$0.0004): five minimap crops
    across ~25 s show the same tile size throughout, the window still at first,
    then sliding north with the player and stopping at the top edge. The route
    reads well. A lava tile beside the stone area was visible and routed around.

### 2026-09-19 — 🎬 checkpoint video final, with the sliding minimap ✅

- `takes.py --seed 21 --map explored`: random 0/5, script 5/5, **Jev 5/5**. All
  102 steps, health 9, 10 calls, 6/10 agreement. **Ralph: "much better" — the
  checkpoint video is accepted.**
- Across three batches that's **15/15 Jev takes with identical decision
  sequences** on seed 21. On this world Jev's choices are stable even though
  its confidence values move between runs.
- Moving on to phase 3.

### 2026-09-19 — phase 3 built: Opus plans + takes over when Jev is unsure

- **`planner.py`:** `claude-opus-5` via the Anthropic SDK (checked against the
  current API docs first). Effort `medium`, adaptive thinking (the Opus 5
  default), and **structured output with `choice` constrained to an enum of the
  menu's keys**, so Opus can't answer off-menu. Refusal fallback `"default"`
  mode is on; if Opus still refuses, is unreachable or over budget, Jev's pick
  stands. `plan()` runs once per episode; `decide()` only on escalation.
  `FakeOpus` for `--dry`.
- **`--agent jev-opus`:** Jev makes every decision. If its top-choice
  confidence is < `--escalate` (default 0.55), Opus decides from the same menu.
  Opus's plan is shown to Jev on every call and listed on the title card.
  Flags: `--escalate`, `--opus-effort`, `--opus-max-calls`, `--no-opus-plan`.
- **Video:** blue banners "Jev unsure (N%): asking Opus" and "Opus overrode
  Jev"; a blue OPUS decision header with Opus's seconds; Opus's reason in blue;
  an amber "Jev leaned to X (N%)". Jev's probability bars stay visible so the
  viewer sees what it was torn between. Cards now wrap long lines, and the plan
  is a numbered list (the "→" glyph rendered as a box, so it was replaced).
- **Price and speed, measured:** Opus ~5 s and ~$0.009 per call, against Jev
  ~0.2 s and $0.000036, so **Opus costs ~250× more per call**. That's why it is
  consulted only when Jev is unsure.
- **First real run, seed 21:** **47 steps** (Jev alone: 102 in all 15 takes;
  script ~49), health 9. One escalation, at step 16 (Jev 23%): Opus overrode
  "collect wood" → "place the table now", and its reason read sensibly. Opus
  total $0.029, 13 s.
  - **Confound:** the plan changed Jev's own behavior too. Alone, Jev chose
    "explore" at step 16 with 91%; with the plan it leaned "collect wood" at 23%,
    which is what triggered the hand-off. A 3-way comparison (plan only /
    hand-off only / both, 11 worlds) is running to separate the two.
- `takes.py`: `--agent jev-opus` (the new default) also runs Jev ALONE
  (`--jev-baseline 3`) so the end card compares random / script / Jev alone /
  Jev + Opus. It checks for `ANTHROPIC_API_KEY` up front. A dry rehearsal passed.
- **Slip, second time:** a cleanup glob `results/*_seed21_base*` removed the
  per-run baseline logs of Ralph's checkpoint run. The totals survive in
  `takes_seed21.json` and in this file, and the Jev take logs are intact. Rule
  saved to memory: my test runs get unique tags, and I delete exact files only.

### 2026-09-19 — phase 3 comparison: plan vs hand-off vs both (11 worlds) ✅

Same 11 explored-map worlds as the Jev-alone evaluation (seeds 21, 1–10), one
run each. **Every configuration reached the goal on 11/11.**

| config | mean steps | median | mean health | Opus calls | Opus $ total |
|---|---|---|---|---|---|
| Jev alone | 53.0 | 44 | 8.9 | 0 | 0 |
| plan only | 38.8 | 35 | 8.7 | 11 | $0.069 |
| hand-off only | 40.5 | **29** | 8.5 | 13 hand-offs (6 overrode Jev) | $0.125 |
| both | 43.4 | 31 | 8.4 | 11 + 15 hand-offs (5 overrode) | $0.238 |

- **Each Opus role on its own captures most of the gain, and they don't
  stack.** "Both" was not better than either alone and cost twice as much.
- **Seed 5 is the outlier, and it's instructive:** hand-off and both took
  145/153 steps against Jev alone's 80. At step 31 a zombie was adjacent and Jev
  was unsure (18%). Opus chose to back away ("break contact first before
  wandering off to hunt for trees"), which is reasonable locally, but it moved
  the player off the route to the trees, and exploring then took ~90 steps.
  Jev alone kept exploring and found trees 5 steps later. **A locally sensible
  decision with a long consequence**, not a bug.
- **Excluding seed 5** (mean steps): Jev 50.3, plan 34.7, **hand-off 30.1**,
  both 32.4. For reference the script on the same worlds was ~31.6. **Hand-off
  alone brings Jev to script-level speed** for ~$0.011 a run.
- Caveats: one run per world, and zombie behavior differs run to run. Health
  was a little lower with Opus (8.4–8.7 vs 8.9), within noise here.
- Added `takes.py --no-opus-plan` so the hand-off-only setup can be recorded.

### 2026-09-19 — takes.py: setups no longer overwrite each other

- Ralph is recording both Opus setups, starting with hand-off only. Found
  before he ran: both would have written `final_jev-opus_seed21.mp4` and the
  same take logs. Output names now carry the setup:
  `final_jev-opus-noplan_seed21.mp4` / `final_jev-opus_seed21.mp4` /
  `final_jev_seed21.mp4`, and matching take and summary files. Checked with a
  dry run; its files were deleted by exact name.

### 2026-09-19 — first hand-off-only recording, and a contaminated baseline

- Ralph's `takes.py --no-opus-plan` run had started before the naming fix, so
  it wrote the old name `final_jev-opus_seed21.mp4`. The setup was right (no
  plan, 1 hand-off per take).
- **Jev + Opus (hand-off only): 5/5, all 38 steps, health 5**, one Opus call
  per take (~$0.011), which overrode Jev each time. The script's median was 54
  in this batch.
- **Contamination, my fault:** my dry check of the new naming ran while
  Ralph's run was going and wrote a FAKE Jev run (`dry=True`, 52 steps) into
  `jev_seed21_base0`, the same log his real run used. His end card's "Jev alone
  3/3 (median 84)" was very likely the median of {52 fake, 84, 102}. **That
  line is unreliable; Ralph is re-running.**
  - Fix: every takes.py log tag now carries the setup (`noplan_`) and dry runs
    carry `dry_`, baselines included, so a rehearsal can't write into a real
    run's files.
- **New fact:** the real Jev-alone runs here were 84 and 102 steps. The 84 run
  skipped one "explore". So the earlier 15/15 identical Jev-alone takes on
  seed 21 were a strong tendency, not a guarantee.

### 2026-09-19 — 🎬 hand-off-only video recorded ✅ (clean re-run)

- `takes.py --seed 21 --map explored --no-opus-plan`, with the new file names:
  random 0/5 · script 5/5 (median 54) · **Jev alone 3/3 (median 104)** ·
  **Jev + Opus 5/5, all 38 steps**, health 5, one Opus call per take
  (~$0.011), which overrode Jev each time.
- **On this world, Jev with the hand-off was faster than the hand-written
  script** (38 vs 54), for about a cent. Ralph: "it was fast; video is fine."
- Video: `videos/final_jev-opus-noplan_seed21.mp4`. Ralph is now recording the
  plan version.
- End-card wording fixed for future runs ("1 call", not "1 calls"), and the
  filmed setup's median steps now appear next to the baselines. The recorded
  video still has the old wording.

### 2026-09-19 — 🎬 plan + hand-off video recorded ✅ — phase 3 done

- `takes.py --seed 21 --map explored`: random 0/5 · script 5/5 (median 49) ·
  Jev alone 3/3 (median 102) · **Jev + Opus 5/5: 44, 52, 47, 44, 44 steps**,
  health 5–9, 2–3 Opus calls per take ($0.019–0.032). Chosen: take 5 (44 steps,
  health 9; Opus's one hand-off agreed with Jev, so 0 overrides; Jev agreed with
  the script on 7/8).
- **The two setups compared on seed 21:**
  - hand-off only: all 5 takes identical, 38 steps, health 5, ~$0.011 per take
  - plan + hand-off: more variable (Opus writes a fresh plan each episode, so
    Jev's context differs), 44–52 steps, better health, ~2–3× the Opus cost.
    In the chosen take the plan did the steering and the hand-off changed
    nothing, which matches the 11-world comparison: each Opus role alone gives
    most of the gain.
- Videos: `final_jev-opus-noplan_seed21.mp4` (hand-off, the cleaner story),
  `final_jev-opus_seed21.mp4` (plan + hand-off), `final_jev_seed21.mp4` (Jev alone).
- This run started before the wording fix, so its end card says "1 hand-offs".
  Cosmetic.

### 2026-09-19 — write-ups ✅

- `FINDINGS.md`: the standalone write-up covering the question, how it works,
  results (11-world table and seed-21 takes), six lessons, the videos,
  caveats, our own mistakes, and open questions. STATUS.md stays the full log.
- `EMAIL_phase4.txt`: a short plain-text follow-up to `../jevsearch/EMAIL1.txt`
  (~230 words, wrapped under 78 columns). Seed-21 numbers are from the same
  batch (script 54, Jev alone 102, Jev + Opus 38).
- Next: the iron-pickaxe run (phase 5).

### 2026-09-19 — iron-run plan written (`PLAN_IRON.md`)

- Re-read Crafter's rules for iron: furnace 4 stone, placeable anywhere; the
  iron pickaxe needs the table AND furnace both within 1 tile; iron needs a
  stone pickaxe; placing stone works on water and lava. Full bill: 5 wood, 5
  stone, 1 coal, 1 iron.
- `seeds.py --iron` over 60 worlds: **seed 21 is still #1** (tree 3, stone 6,
  coal 7, iron 10, water 7, no lava within 12).
- **Gap found while planning:** today's reachability requires open ground
  beside the target, and iron is usually walled in by stone, so the current
  code would treat most iron as unreachable. Dig-aware reachability is the
  first item in the plan.
- Steps I-0…I-4: groundwork + scripted iron agent (free, 40 worlds) →
  survival/night (free) → Jev alone (11 worlds) → Jev + Opus (11 worlds) →
  record. Five open questions for Ralph at the end of the plan.

### 2026-09-19 — iron plan approved; I-0 started

- Ralph approved Claude's answers to the 5 questions: both crafting-spot
  options on the menu · shelter/sleep only if needed · 8 fps · seed 21 ·
  record hand-off only unless the plan proves itself on iron. Recorded in
  `PLAN_IRON.md`.

### 2026-09-19 — I-0 part 1 (session restarted mid-step; verified intact)

- Built:
  - `approachable()`: a stand-spot counts if you can walk there OR dig into it
    with the tools held
  - dig-aware `Collect` goal and `Explore` frontier (they can tunnel into
    mountains)
  - `GoNear` with several materials (the table + furnace spot)
  - `Place(..., keep_near=...)` so the furnace lands beside the table
- The session restarted after this edit. Checked afterwards: all modules
  parse, `-h` works, and the **stone run is unaffected** (scripted, explored
  map, 11/11 worlds). The dig-aware change applies to the stone run too; logged
  here in case its step counts ever shift.
- Still to do in I-0: generic goal resolver (`remaining()` is still
  stone-only), iron menu entries (coal, iron, furnace, crafting spot, second
  table), scripted iron agent, `--goal iron_pickaxe`, the 40-world measurement.

### 2026-09-19 — I-0 part 2: iron goal wired end to end

- Phase 4 marked done in the checklist (Ralph: the 11-world evaluations plus
  `EMAIL_phase4.txt` completed it).
- **Generic goal resolver** `remaining()`: walks Crafter's recipe tables
  (ingredients, the table/furnace a tool is crafted beside, and the pickaxes
  needed to mine the ingredients), counting each tool and utility once and
  skipping anything held or placed. Checked against 6 hand-worked cases,
  including partial progress (stone pickaxe held → 1 wood, 4 stone, 1 coal, 1
  iron; furnace placed → the 4 stone drops off).
- **Iron menu** (the stone menu is unchanged): mine coal, mine iron, place the
  furnace ("beside the table" when the table is in reach), walk to the table
  and furnace, and a **second table** option offered only when it leaves
  enough wood for the goal. It gives the first table's distance in words.
  `reachable()` is now dig-aware. Coal and iron count as "wanted" for exploring
  only once you hold the pickaxe that mines them.
- **Scripted iron agent:** the same recipe up to the stone pickaxe, then
  coal → iron → any short stone/wood → walk back to the table → furnace beside
  it → craft. It never takes the second-table option; that's left for Jev/Opus.
- `run.py --goal iron_pickaxe` (900 steps, 8 fps by default), plus a **cause
  of death** in every summary (lava / zombie / no food, drink or energy, with
  time of day and step).
- **First runs, seed 21: iron pickaxe reached on the first try.** Full map 105
  steps, explored 96, no skill failures. It dug through stone to coal and iron
  (stone piles up in the inventory while tunnelling). **The whole run fits
  before dusk (~150)** on this world, so night may matter less than planned,
  at least for the script here.
- A 40-world measurement (iron and stone, full and explored, ~160 free runs)
  is running.

### 2026-09-19 — EMAIL_phase4: why Opus was asked only once

- Ralph asked for the reason. From the recorded run's log: Jev was confident
  on 7 of 8 decisions (67–97%); hand-off only happens below 55%. The one
  hand-off came at step 25, when a zombie appeared just as the table could be
  built. Jev wavered at 45% toward backing away, and Opus said to ignore it
  briefly and build the table. That's the same moment where Jev alone runs off
  (~100 steps). Added as a short paragraph, in plain words.

### 2026-09-19 — I-0 measured: the script reaches iron, and night is the killer

Scripted agent, 40 worlds each (seeds 1–40), one run per world:

| goal | map | reached | median steps | failures |
|---|---|---|---|---|
| iron pickaxe | full | **36/40** | 94.5 | 3 zombie deaths at night (steps 187–200), 1 out of energy (step 309) |
| iron pickaxe | explored | **25/40** | 89 | 15 deaths, **all at nightfall or after** (steps 150–250): mostly zombies, 2 also out of food, 3 "unknown" (likely skeleton arrows) |
| stone pickaxe | full | 40/40 | 30 | none: **stone run unchanged** by the iron work |
| stone pickaxe | explored | 40/40 | 32 | none |

- **Seed 21, explored, 5 runs: 5/5**, 92–96 steps, all before dusk.
- **The pattern is sharp.** Successful iron runs finish around step 90, before
  night. Runs that go past ~170 steps meet the night and die. The script has no
  night behaviour at all: it keeps working and fights zombies one at a time,
  bare-handed (5 hits each).
- I-0's target (≥80% on the explored map) is **not met: 62%**. The gap is
  night survival, which is I-1's job, and the plan's condition for building
  shelter + sleep ("only if I-0 shows need") is clearly met.
- Proposed for I-1, cheapest first:
  1. **Swords.** A wood sword (1 wood) or stone sword (1 wood + 1 stone) at the
     table cuts a zombie from 5 hits to 3 or 2. Not in the original plan.
  2. **Shelter + sleep at night:** wall in with stone (runs are often carrying
     5–9 spare stone) and sleep until dawn. That also fixes the one energy death.
  3. **Eat earlier** (food ≤ 3 rather than ≤ 2).

### 2026-09-19 — I-1 built: swords, shelter, eat earlier, night video, iron-aware Jev

- **Shelter needs a trick.** Crafter only places a block on the tile you face,
  and you can only face open ground by stepping onto it, so "step into a hole
  and seal the gap behind you" is impossible. The solution is a **two-tile
  room**: dig A and B straight into rock, step into B, step *back* into A (now
  facing the entrance), place stone. `find_shelter_site()` looks for an entrance
  with rock on every side of A and B and beyond. `Shelter` then sleeps (or
  waits) until daylight > 0.6.
- **Menu (iron goal only):** swords, offered only from *spare* materials, with
  the hit counts worked out in code ("zombies then take 2 hits instead of 5");
  and "shelter", offered only while night is coming or here (steps 130–250 of
  each 300-step day), not at dawn when daylight is also low.
- **Script switches** (each can be turned on separately for the measurement):
  `swords` (collect +1 wood and +1 stone, craft a sword at the table), `shelter`
  (top up drink/food, then dig in at night), `eat_early` (act at 3, not 2).
- **Seed 34 explored**, a night death in I-0, now reaches iron in 99 steps
  (wood sword at step 28).
- **Video:** night brightness floor (`NIGHT_LIFT` 0.8, i.e. up to 1.8×, to
  tune on real night frames); the panel shows the top 8 of larger menus.
- **What Jev sees** for iron: progress for all three pickaxes, coal and iron,
  furnace placed, table and furnace both in reach, sword held; distances to
  coal, iron and furnace; `lava_on_screen`; iron rules added to the rules text.
- Measurement running: I-0 baseline → + swords → + eat early → + shelter, on
  40 explored worlds, plus everything on the full map.

### 2026-09-19 — EMAIL_phase4 sent

- Ralph sent `EMAIL_phase4.txt` (with his own edit to the "why only once"
  paragraph), adding a note that the iron run is in progress. Claude offered
  a four-line paragraph for that, consistent with the I-0 numbers.

### 2026-09-19 — I-1 round 1 measured: the shelter works, then hunger bites

Scripted iron agent, 40 worlds, each change added on top of the last:

| setup | explored map | full map |
|---|---|---|
| I-0 baseline | 26/40 | (36/40 in I-0) |
| + swords | 24/40 | |
| + eat earlier | 25/40 | |
| + shelter (all three) | 26/40 | **40/40** |

- **Full map: 40/40** with everything on. All 6 runs that sheltered went on to
  reach the iron pickaxe.
- **Explored map: no net change**, and the failures moved rather than
  vanished. Zombie deaths fell (11 → 5), but **starvation rose to 6**. The
  runs survive the night, then starve the next day, because exploring for coal
  and iron makes explored runs long (300–600 steps) and a cow is often not in
  sight.
- Swords made no measurable difference (24–26/40 is within the noise).
- Re-running two night deaths showed why:
  - **seed 39:** sheltered at step 166, came out at 281 with **health 9**, and
    reached the goal, **so the shelter works**, but it nearly starved afterwards.
  - **seed 40:** started digging at step 177, already dark, and was
    interrupted by fights until it died. **The "night is falling" interrupt
    from the plan was never built**, so a long explore ran straight through
    dusk.
- Round 2 fixes:
  1. the **night-falling interrupt** (once per night, iron goal)
  2. a **"search for a cow"** option when food ≤ 4 and none is in sight, plus
     eating a visible cow at food ≤ 5
  3. the shelter **fights a zombie standing in the doorway**, then seals
  4. zombies no longer interrupt a shelter being dug
- Round 2 measurement is running.

### 2026-09-19 — I-1 round 2 measured: worse overall, and the noise is large

| setup (40 worlds, one run each) | explored | full |
|---|---|---|
| round 1 (swords + eat earlier + shelter) | 26/40 | 40/40 |
| round 2 (+ night interrupt, find-cow, doorway fight, no zombie interrupt while digging) | **23/40** | **34/40** |

- **Starvation is fixed** (6 → 1 on explored), but **zombie deaths went up**
  (5 → 12 explored; 0 → 6 full).
- The **noise is large**. One full-map death came at step 33 in daylight (seed
  16, a world that already gave the stone run trouble with zombies), which
  none of the round-2 changes touch. With one run per world, a swing of ±3/40
  is well within chance. The 40 → 34 full-map drop may be partly real (a
  likely suspect: zombies can no longer interrupt a shelter being dug, so the
  player digs while being hit), but this data can't separate the causes.
- Several explored night deaths never sheltered at all (seeds 12, 14, 26, 30,
  37, 38). Likely no known rock pocket with solid sides, or no stone in hand
  at dusk, but not verified.
- **Seed 21, current code: 5/5**, 100–101 steps, health 6–8, all before
  night. The demo world is unaffected.
- **Recommendation:** stop tuning the script's night survival here. It is the
  *baseline*, not the demo, and further gains need multi-run measurement to
  see through the noise. Its honest record is "reaches iron on ~60–65% of
  explored worlds, and dies at night when it doesn't". That also makes a real
  question for I-2/I-3: **can Jev and Opus handle the night better than the
  script?** Awaiting Ralph.

### 2026-09-19 — the shelter-interrupt check: no effect, and the noise is now measured

40 full-map worlds, same script, one run each, only one thing changed:

| may a zombie interrupt a shelter being dug? | reached iron | median steps | sheltered (then reached goal) |
|---|---|---|---|
| no (current) | **36/40** | 96 | 12 (9) |
| yes | **36/40** | 94.5 | 14 (10) |

- **No difference.** The change is not what caused round 2's drop.
- **The important part:** 8 worlds changed outcome between the two arms, and
  **no world failed in both**. The 4 failures in each arm are disjoint sets.
  With one run per world, which worlds fail is close to a coin toss.
- That puts the earlier numbers in perspective: round 1's 40/40 and round 2's
  34/40 on the full map are both consistent with **~36 ± 3**, i.e. the same
  script. **None of the I-1 comparisons had the resolution to rank the
  variants.** What is solid: the script reaches iron on most full-map worlds
  and about 60% of explored ones, night is what kills it, and the shelter
  works when it is dug in time (9–10 of 12–14 shelter runs reached the goal).
- Code left as is (zombies do not interrupt digging); there is no evidence
  either way.
- **Mistake on my side:** the first attempt at this check never ran. My edit
  inserted the new constant into three other places that mention
  `COST_PER_CALL`, so `run.py` stopped parsing. The error went to the
  background log, not to me. Nothing was lost (it failed before any runs), and
  the lesson is to parse-check a file immediately after editing it, which I
  now do.

### 2026-09-19 — HANDOFF written; session about to restart

- Ralph asked for a handoff before continuing. Wrote `HANDOFF.md`: one page
  covering what the project is, the five findings that shape the design, where
  the stone and iron runs stand, the hard-won lessons (measurement noise,
  Crafter's non-reproducibility, never call `env.render()`, the facing rule,
  parse-check after editing, no glob deletes), the conventions, the commands
  and the file map.
- Added a memory pointer so a new session reads `HANDOFF.md` first.
- **State at handoff:** phases 0–4 done (stone videos, findings, email sent).
  Phase 5: I-0 and I-1 done; script reaches iron on ~36/40 full and ~25/40
  explored worlds, night being the usual cause of failure; seed 21 succeeds
  every time in ~100 steps, before dusk.
- **Next:** I-2, Jev alone on iron, 11 worlds, ~$0.01. Optional first: the
  cheap check of whether "shelter" is even offered on worlds that died at
  night without sheltering.

### 2026-09-19 — the shelter question answered, the priority fixed, and I-2 measured

**1. Was "shelter" ever offered on the worlds that died at night?** Yes, and
that was the surprise. Re-running the six explored worlds that died at night
without sheltering (12, 14, 26, 30, 37, 38):

- seed 38 was offered shelter at steps 140-218, **nine times**, with 9 stone in
  hand, and took none of them. It spent the whole night on `find_cow` (no cow
  in sight) and `fight_zombie`, and died at 224.
- seed 37 was offered five times, took it at 212 — after dark — and still died.
- seed 30 was offered once at 178, behind a `fight_zombie`.
- seeds 12, 26 finished before dusk this time and 14 sheltered and reached the
  goal: **the coin-toss noise again**, three of six worlds no longer died.

So the models can be judged on night decisions: the option is on the menu.

**2. The cause was a priority bug in the scripted agent**, not the menu. The
shelter rule sat *below* `fight_zombie`, the thirst/hunger criticals and
`find_cow`, so anything else at all outranked digging in. Fixed in `agents.py`:
the night block now runs first, and only an adjacent zombie at health <= 4
comes before it (the shelter skill handles one in the doorway itself). The old
order is kept behind `ScriptedAgent.shelter_first = False` for the measurement.

**3. A/B on the fix** — 40 worlds x 2 arms x 2 runs = 160 explored runs, 8-way
parallel:

| | old order | new order |
|---|---|---|
| reached iron | 49/80 (61%) | 50/80 (62%) |
| median steps | 99 | 99.5 |
| **deaths at night** | **20** | **9** |
| starvation | 5 | 11 |
| shelter offered -> declined | 199 -> 161 (81%) | 93 -> 46 (49%) |
| runs that sheltered | 25 | 36 |

- The mechanism works: night deaths less than half, and the script now takes
  the option about twice as often. The declines that remain are the block's own
  top-up branches (drink/eat before a long night) and the health <= 4 fight.
- **Success is unchanged**: paired, new-only wins 7, old-only wins 6 of 80.
  Failure *moved* — survive the night, starve the next day — exactly the
  conservation seen in I-1 round 1. Starvation is now the top cause.
- Fewer offers in the new arm because a run that shelters at the first offer
  never sees the next eight.
- Kept on: it removes a real misbehaviour and makes night decisions legible,
  but it does not lift the script's ~61%.

**4. I-2 — Jev alone on iron**, 11 explored worlds (seeds 21, 1-10), run twice:
once with the old script order, once after the fix (the fix changes only the
script, so Jev's second arm is a fresh sample, not a corrected one).

| run | agent | reached | median steps | mean health | agreement | cost |
|---|---|---|---|---|---|---|
| I-2 (old order) | scripted | 8/11 | 100.5 | 8.4 | - | free |
| I-2 (old order) | **Jev** | **6/11** | 101 | 7.7 | 95/191 (50%) | $0.0066 |
| I-2b (shelter first) | scripted | 8/11 | 100 | 8.6 | - | free |
| I-2b (shelter first) | **Jev** | **7/11** | 119 | 8.4 | 85/205 (41%) | $0.0071 |

- **Jev is roughly level with the script, maybe a shade behind**: 6/11 and 7/11
  against 8/11 twice. A 1-2 world difference on 11 worlds is noise.
- **Agreement fell to 41-50%**, against 85% on the stone run. The iron menu is
  much bigger, so there are many more ways to disagree; this is the number to
  watch in I-3, since it is where Opus's hand-off should earn its keep.
- **Seeds 5, 8 and 9 failed in every arm.** Unlike the I-1 A/B, where no world
  failed in both arms, these three are genuinely hard worlds, not coin tosses.
- Seed 21 (the demo world) reached iron in all four arms: script 100-101 steps,
  Jev 74 and 291.
- Logs kept: `results/{scripted,jev}_seed*_i2*.jsonl`.

### 2026-09-19 — I-3: the hand-off is what works on the long task ✅

Three Opus setups, the same 11 explored worlds (seeds 21, 1-10), one run each,
4 episodes in parallel:

| setup | reached iron | median steps | mean health | Opus calls | overrides | cost |
|---|---|---|---|---|---|---|
| plan only (no hand-off) | **4/11** | 92.5 | 7.2 | 11 | 0 | $0.10 |
| hand-off only | **10/11** | 91.5 | 7.7 | 72 | 44 | $0.71 |
| plan + hand-off | **11/11** | 106 | 7.2 | 109 | 49 | $1.20 |

For reference on the same worlds: **scripted 8/11**, **Jev alone 6/11 and
7/11**.

- **This is the first time the models clearly beat the hand-written script.**
  On the stone run the script was hard to beat; on the long iron task the
  hand-off wins 10-11/11 against 8/11, and it solves the worlds nothing else
  could.
- **Seeds 5, 8 and 9 failed in every script and Jev-alone arm.** Hand-off
  cleared 5 and 8; plan + hand-off cleared all three. Those were the worlds
  called "genuinely hard" in I-2, so this is the clearest signal in the set.
- **The plan alone is worse than no plan at all** (4/11 against Jev's 6-7/11).
  With `--escalate 0` it never hands off, so it is Jev reading a fixed plan
  written before step 0 — and a plan written before the world is known goes
  stale. Six of its seven failures are starvation or a day-time zombie at step
  ~300, i.e. it kept following the recipe while the situation changed.
- **Opus overrides Jev on 61% of the decisions it is asked about** (44 of 72
  hand-off calls). That is much higher than the stone run, where it was
  consulted once. The bigger iron menu produces more low-confidence moments.
- **Cost ran over the estimate**: $2.01 for the three setups, against the
  planned $0.30-0.60, because the iron menu triggers many more escalations
  (72-109 Opus calls, not the ~3-5 per episode assumed). Per world: hand-off
  $0.065, plan + hand-off $0.11.
- **For the video (decision 5), hand-off only still wins.** The extra world
  that plan + hand-off reaches is within noise, it costs 70% more, and on seed
  21 itself it is far slower: hand-off 76 steps, plan + hand-off 336.
- Logs: `results/jev-opus_seed*_i3_*.jsonl`.

### 2026-09-19 — I-4: the iron video is recorded ✅

`takes.py --seed 21 --map explored --goal iron_pickaxe --no-opus-plan`, served
`jev-1.13.0`, hand-off only:

- baselines on seed 21: **random 0/5, script 5/5** (median 100 steps), **Jev
  alone 3/3** (median 109)
- **Jev + Opus 5/5**: 76, 77, 80, 76, 76 steps — **the whole set beats the
  script's 100**, which the stone run only managed with Opus's help too
- takes cost $0.059-$0.092 each (6-9 hand-offs, 3-5 of them overriding Jev)
- chosen automatically: **take 3** (80 steps, health 7 — the health tiebreak
  beat three 76-step takes at health 6). `--pick 1` would take a 76-step one.
- **`videos/final_jev-opus-noplan_iron_seed21.mp4`**, 39 s, 1440x864, 8 fps,
  decodes clean.

**Two fixes `takes.py` needed first:**

1. **It had no `--goal`** — it was hard-wired to the stone pickaxe, although
   PLAN_IRON's I-4 command already assumed the flag. Added, with the goal in
   the variant name so iron output cannot overwrite the stone videos, and the
   **end card now renders at the take's fps** (a 6 fps card concatenated onto
   an 8 fps take is how you get a silently broken file).
2. **A dry rehearsal shared take filenames with a real run.** The `dry_` prefix
   guarded the result logs but not the MP4s, so a rehearsal would have
   overwritten real takes - the same class of bug as the earlier dry-run log
   collision. Dry runs now carry `_dry` through the whole variant name.

A free `--dry --takes 2` rehearsal ran first and produced a truthful "0 of 2
reached the goal" card, which is what caught both.

**Session spend:** $2.42 (I-2 $0.007, I-2b $0.007, I-3 $2.01, I-4 $0.42).

### 2026-09-19 — I-3 second pass: the result holds, and plan + hand-off never failed

The same 33 episodes again (3 setups x 11 worlds), pooled with the first pass —
**22 runs per setup**, which is what it takes to see past this phase's noise:

| setup | pass 1 | pass 2 | **pooled** | median steps | Opus calls | cost |
|---|---|---|---|---|---|---|
| plan only (no hand-off) | 4/11 | 6/11 | **10/22** | 106.5 | 22 | $0.19 |
| hand-off only | 10/11 | 9/11 | **19/22** | 96 | 141 | $1.47 |
| plan + hand-off | 11/11 | 11/11 | **22/22** | 106 | 200 | $2.20 |

On the same worlds: **scripted 16/22** (8/11 twice), **Jev alone 13/22** (6/11
and 7/11).

```
per-world, both passes      21    1    2    3    4    5    6    7    8    9   10
plan only                   XX   oo   Xo   oo   oX   XX   oo   Xo   XX   XX   Xo
hand-off only               oo   oo   oo   oo   oX   oo   oo   oo   oo   XX   oo
plan + hand-off             oo   oo   oo   oo   oo   oo   oo   oo   oo   oo   oo
```

- **Plan + hand-off reached the iron pickaxe in all 22 runs**, including seeds
  5, 8 and 9, which failed in every scripted and Jev-alone arm. That is the
  headline, and it is now on 22 runs rather than 11.
- **The hand-off is doing the work.** 19/22 on its own against the script's
  16/22; seed 9 beat it both times and seed 4 once.
- **Plan + hand-off vs hand-off alone is 3 runs of 22** — suggestive, not
  established, and it costs 50% more ($0.10 an episode against $0.067).
- **A plan on its own is not worth having: 10/22, against Jev alone's 13/22.**
  Pass 2 (6/11) was kinder than pass 1 (4/11), so "worse than no plan" is not
  proven — but two passes agree it does not help, and its failures are the same
  shape both times: starvation or a day-time zombie around step 300, the agent
  still following a recipe written before it had seen the world. **Yet the same
  plan, combined with the hand-off, is what turns 19/22 into 22/22.** The plan
  is not useless; it is useless *without someone to notice it has gone stale*.
- Opus overrode Jev on **60%** of hand-off calls (89 of 148) with no plan, and
  **44%** (78 of 178) when a plan was present — the plan makes Jev's low-
  confidence moments less often wrong.
- **The video choice does not change.** On seed 21 hand-off only is 76 steps in
  both passes; plan + hand-off took 336 and 112. `I-4` stands as recorded.
- Pass 2 cost $1.86. Logs: `results/jev-opus_seed*_i3b_*.jsonl`.

**Session spend: $4.28.**

### 2026-09-19 — the iron write-up and a README for sharing ✅

- **`FINDINGS.md` now covers both runs.** The stone material is unchanged and
  became **Part 1**; **Part 2** is the iron pickaxe: why a longer goal, the
  22-runs-per-setup table, the seed-21 takes, six findings, the video, caveats,
  our own mistakes, and a refreshed set of open questions. The header tells a
  reader to start at Part 2's results. Part 1's old open questions became a
  short "what Part 1 left open" bridge, since the first of them is now answered.
- Part 2's six findings: the hand-off is worth more the longer the game; a plan
  alone is the worst setup while plan + hand-off is the only perfect one;
  disagreement (85% → 50% agreement) is where the value is; fixing a survival
  failure moves it rather than removing it; the bug was in our baseline, not the
  model; and measurement noise was the hardest part of the run.
- **`README.md` written for friends and for a future repo:** what it is and the
  headline table, a quick start that works with no keys at all (`--dry`), the
  three keys and what needs them, how Jev and Opus fit together (the 0.55
  hand-off, the plan, and the two design rules), then **each runnable program**
  — `run.py`, `takes.py`, `seeds.py` — with an options table, examples and its
  outputs, then the supporting modules one row each, the output layout, the
  documentation map, the reproducibility caveats and a cost table.
- It also flags the one thing that would break a copied repo: `chooser.py`
  imports the Jev client from `../jevsearch/client.py`.
- **Checked rather than asserted:** the quick-start command runs clean (50
  steps, a 27 s MP4, 1.2 s of compute), and `seeds.py` does rank seed 21 first
  for both goals, as the README claims.

### 2026-09-19 — HANDOFF rewritten for tomorrow; day closed

- `HANDOFF.md` now opens with **"Tomorrow: two things, both small"**, ahead of
  everything else, so a cold session starts on the work rather than the history:
  1. **a new email file** (`EMAIL_phase5.txt`), with `EMAIL_phase4.txt` as the
     model and Part 2 of `FINDINGS.md` as the material — plus the two questions
     to ask Ralph (same recipient? video linked or attached?);
  2. **a clean directory for his git repo, which lives elsewhere on disk** —
     the exact copy list (9 modules, 4 config files, 5 documents, 4 finished
     videos), what must *not* be copied (`rmb_*`, `.venv/`, `__pycache__/`,
     `results/`, `videos/takes/`, `zz_*`), and the reminder that the existing
     `.gitignore` already excludes `videos/`, so the 4.5 MB of MP4s need
     `git add -f` if they are wanted in the repo at all.
- **The one real decision recorded for that task:** `chooser.py:26` reaches into
  `../jevsearch/client.py`, which a standalone `jevcraft` repo will not have.
  Either copy the client in and change the import, or ship `jevsearch/`
  alongside — Ralph's call, and the only thing that would stop a fresh clone
  from running.
- **Pre-publication check run and passed:** no API keys, tokens or email
  addresses in any shippable file; the only absolute `/Users/rbutler` paths are
  in `HANDOFF.md`, which is deliberately not on the copy list. Every file on the
  list was confirmed to exist.
- The middle of the HANDOFF was compressed (the per-step iron detail now lives
  in `FINDINGS.md` Part 2) and the leftover "open choices for the write-up"
  became two judgement calls for the email.
- **Day's totals:** phase 5 finished end to end (I-2, I-3 twice, I-4), the night
  priority bug found and fixed, `FINDINGS.md` Part 2 and `README.md` written,
  $4.28 of API spend, nothing left running.
