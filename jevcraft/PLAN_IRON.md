# ⛏️ jevcraft — the iron-pickaxe run (phase 5)

**Goal:** same setup as the stone run (explored-only map, Jev picks, Opus takes
over when Jev is unsure, video with the side panel), but the target is an
**iron pickaxe**. That's a longer recipe, iron is buried inside mountains, and
the game runs long enough for thirst, hunger and night to matter.

**Why it's worth doing:** on the stone run a hand-written script is hard to
beat, because the recipe is short and fixed. The iron run is where rigid
recipes tend to break (a zombie mid-tunnel, thirst at step 180, night) and
where judgment, Jev's and Opus's, has real work to do.

**Status (2026-09-19): all five steps done.** I-0 script, I-1 survival/night,
I-2 Jev alone (6-7/11), I-3 Jev + Opus (**10-11/11** against the script's 8/11),
I-4 the video (`videos/final_jev-opus-noplan_iron_seed21.mp4`). Two things
changed against this plan as written: the crafting-spot second-table option was
built but the script never takes it, and shelter/sleep was built *and* had its
priority fixed, which halved night deaths without moving the success rate.
`STATUS.md` has the numbers.

---

## 🔎 Verified facts (Crafter 1.8.3 source, read again for this plan)

| fact | consequence |
|---|---|
| Table 2 wood · furnace **4 stone** · placing needs no nearby table | the furnace can go anywhere |
| Iron pickaxe: 1 wood + 1 coal + 1 iron, **table AND furnace both within 1 tile** | the crafting spot must have both next to the player |
| Coal needs a wood pickaxe; **iron needs a stone pickaxe** | a strict order: wood → table → wood pickaxe → stone → stone pickaxe → coal, iron, furnace |
| Full bill: **5 wood, 5 stone, 1 coal, 1 iron** | twice the stone-run bill, plus two new materials |
| Iron and coal sit **inside** mountains, usually with no open ground beside them | the agent must **dig** to reach them |
| Placing stone works on **water and lava** too | stone can bridge water or cap lava |
| Drink drops every ~20 steps, food ~25, energy ~30, all from 9 | a run past ~180 steps must drink; past ~270, energy runs out |
| Night: dusk ~150, dark ~180–240 of each 300-step day; zombies spawn in the dark and hit a sleeper for 7 | the iron run will probably meet the night |
| Seed ranking (`seeds.py --iron`): **seed 21 is still #1**: tree 3, stone 6, coal 7, iron 10, water 7, no lava within 12 | keep seed 21 as the demo world |

---

## 🧰 What we reuse vs what is new

**Reused as is:** pathfinding (it already digs through stone once the right
pickaxe is held), `Collect` (coal and iron are generic materials), `Place`,
`Make`, `GoNear`, `Flee`, `Attack`, `Explore`, drink and eat, the Jev and Opus
agents, the renderer, `takes.py`.

**New or changed:**

1. **Goal resolver.** `remaining()` is hard-wired to the stone pickaxe. Replace
   it with a small resolver that reads Crafter's recipe tables and returns
   what is *still* needed for any goal: tools, placed things, raw materials.
   Menu labels stay "have 2, 3 more needed". All arithmetic stays in code.
2. **Dig-aware reachability (the big one).** Today "reachable" means open
   ground next to the target. Iron is usually walled in by stone, so it would
   look unreachable forever. The fix is to count a stand-spot you can **dig
   into** (with the right pickaxe) as reachable, for `reachable()`, `Collect`
   and `Explore`. Explore should also treat diggable stone at the edge of the
   known map as a frontier, since mountains are where iron is.
3. **The crafting spot.** The iron pickaxe needs the table and the furnace
   *both* within one tile. Two ways to get there, and I'd put **both on the
   menu** so choosing is a real decision:
   - *walk back and place the furnace beside the existing table*: no extra
     wood, but maybe a long walk
   - *build a second table next to the furnace, near the iron*: costs 2 wood,
     no walk back
   The code states the trade-off in words ("the table is far; a second one
   costs 2 wood").
4. **Survival for a long game.**
   - Drink and eat get urgency words in the menu ("thirst low, water is
     close") and a firm rule in the script.
   - **Shelter + sleep:** a skill that walls the player in with stone (4
     stone), then sleeps, so zombies can't reach a sleeper. Only needed if
     runs pass ~270 steps; build it only if the measurements say so.
   - Lava: never walked on, and new, a "lava nearby" warning in the situation
     Jev sees.
5. **Night on video** (decision 7, already agreed): a brightness floor applied
   to the frame at render time (the game still sees true night), plus the
   existing NIGHT label. No extra `env.render()` calls, since those change
   zombie behaviour.
6. **Panel for a bigger menu.** The iron menu can reach 12+ options; the panel
   shows 8. Show the 8 most likely by Jev's probabilities, and always the one
   chosen.
7. **Longer video.** ~300 steps plus ~25 held decisions is about 2 minutes at
   6 fps. Options: 8 fps, or a shorter hold (1.0 s).

---

## 🪜 Steps

Each step ends with something measurable, and steps I-0 and I-1 cost nothing.

### I-0 — groundwork + scripted iron agent *(no API)*
- goal resolver, dig-aware reachability, crafting-spot options, script rules
- **measure:** scripted success on 40 worlds, full map and explored map
- **done when:** the script reaches the iron pickaxe on most worlds (target
  ≥ 80% explored) and on seed 21 every time, with steps and cause of death
  logged for every failure

### I-1 — survival and night *(no API)*
- drink/eat rules, the lava warning, the night brightness floor
- shelter + sleep **only if** I-0 shows runs dying of energy or night
- **done when:** failures in I-0 caused by thirst, hunger or night are fixed,
  or explained in the log

### I-2 — Jev on iron
- `--dry` first, then live; 11-world evaluation, Jev alone (~$0.01 total)
- **done when:** we know Jev's success rate and steps against the script's

### I-3 — Jev + Opus on iron
- hand-off only, and plan + hand-off, on the same 11 worlds (~$0.30–0.60, more
  Opus calls than the stone run)
- **done when:** the same four-way comparison table exists for iron

### I-4 — record
- `takes.py --goal iron_pickaxe` on seed 21: baselines, then 5 takes, best-of-5
  end card
- **done when:** there is an iron-pickaxe video you're willing to show

---

## ⚠️ Risks

| risk | mitigation |
|---|---|
| Iron found late or not at all on the explored map | dig-aware frontier in Explore; seed 21 has iron 10 steps out; log every failure's cause |
| Death at night, or by lava while digging | lava never pathable; flee/fight menu; shelter if I-0 shows night deaths |
| Runs long enough that energy runs out (~270 steps) | measure first; shelter + sleep is the fix if needed |
| Menu too big to read | top-8 display; every option still goes to Jev |
| The script turns out to be as good as Jev + Opus again | fine, we report it. But the iron run is where we expect a gap if there is one |
| Opus cost per run grows | per-episode cap (`--opus-max-calls`), and hand-off only was the cheaper setup on stone |

---

## 💰 Cost

- I-0, I-1: free
- I-2: ~11 games × ~20 Jev calls ≈ $0.01
- I-3: ~22 games × ~3–5 Opus calls ≈ $0.30–0.60
- I-4: ~5 takes + baselines ≈ $0.10–0.20

---

## ✅ Decisions (2026-09-19, Ralph: "go with your answers")

| # | question | decision |
|---|---|---|
| 1 | crafting spot | **both options on the menu**: walk back vs a second table |
| 2 | shelter + sleep | **only if I-0 shows it is needed** |
| 3 | video length | **8 fps** for the iron run |
| 4 | demo world | **seed 21** |
| 5 | Opus setups to record | **hand-off only**, unless the iron comparison shows the plan helping |
