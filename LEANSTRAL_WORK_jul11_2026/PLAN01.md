# 🧬 Leanstral 1.5 Demo Plan (PLAN01)

## 🎯 Goal
Use **Leanstral 1.5** (4-bit quant, local via LMStudio) to do a *meaningful* formal-verification
solve in **Lean 4**, then write it up for colleagues (email now; possibly a rehearsed video later).
Bar: must be genuinely meaningful, not "look, it does something."

## 👥 Audience
Mixed lab: biologists-who-code, CS folks with classical ATP experience (E-prover et al.),
and students learning. All have math background; none claim to be mathematicians.
→ Direction: **code verification**, not abstract math. Bio-flavored targets.

## 🧑‍🔬 Working mode
- **Not** primarily a live demo. It's **RB + Claude doing a real solve together**, then writing it up.
- Claude leads and runs the Lean toolchain + model calls; RB narrates / presents.
- RB is new-ish to Lean 4 (finds it opaque); comfortable with TPTP/E-prover. We've done Lean together before.
- Deliverable = **the writeup (email)**. Optional later: rehearse once, then record a video.

## ⏱️ Budget
- Regular attempts: **under ~10 min** each.
- Reserve **one ~40-min ambitious run** for the meaty arc (target A's full find-and-fix).
- Pre-bake a proven fallback so nothing hard-fails; video allows pausing during runs.

## 🧪 Targets (chosen: B then A)
Both are real, documented classes of comp-bio bugs (credibility for the email).

### B — Reverse-complement round-trip (involution)  *[warmup, safest first win]*
- **Spec:** `revcomp(revcomp(s)) = s`; maps A↔T, C↔G while reversing.
- **Bug:** IUPAC ambiguity codes (R↔Y, …) or soft-masked lowercase bases handled wrong → round-trip silently breaks on real reference sequences.
- **Counterexample:** sequence containing `R` (or a lowercase masked base) → round-trip ≠ input.
- **Why:** elegant involution spec (ATP crowd appreciates it); universally recognized op; lowest Lean difficulty → fast clean win to warm up the toolchain.

### A — 0-based/1-based interval overlap (off-by-one)  *[the real story]*
- **Spec:** intervals overlap iff `a.start < b.end ∧ b.start < a.end` (BED half-open).
- **Bug:** classic `a.end >= b.start` (`>=` on half-open coords) → reports *adjacent* features as overlapping.
- **Counterexample:** exon `[10,20)` vs feature `[20,30)` — don't overlap, buggy check says they do; a variant at pos 20 annotated to the wrong gene.
- **Why:** *the* bug every comp-bio person has hit (BED vs GFF/VCF/SAM conventions); obvious real stakes; films with zero setup.
- **Arc:** find bug → concrete counterexample → fix → **prove correct for all inputs** (the hero step). This gets the 40-min budget if needed.

### C — CIGAR reference-span  *[backup if A gets fiddly]*
- **Spec:** ref length = sum of ref-consuming ops (M/D/N/=/X), excluding I/S/H/P.
- **Bug:** counting insertions (`I`) toward ref length → wrong alignment end → downstream overlaps off.
- **Counterexample:** CIGAR `5M3I5M` → true span 10, buggy says 13.

## 🧠 Why Lean at all? (the email's thesis — must preempt the skeptic)
Concede up front: a modern LLM usually *catches* B and A by eye. The value isn't the catching:
1. **Certificate vs. guess** — kernel-checked proof vs. a probability; the model that wrote the bug shares the coder's blind spot, the kernel doesn't.
2. **∀ vs. sampling** — the *fix* is the hero: proven for **all** inputs, not "looks right on cases I tried" (tests/fuzzing/LLM-eyeballing all sample).
3. **Can't hallucinate a proof** — a false property just fails instead of producing a confident wrong argument.
- B and A are eyeballable *on purpose*: audience can verify the tool isn't cheating. Same machinery scales to code no one can eyeball (paper's AVL `O(log n)`, 2.7M tokens).

## 🎭 Anti-hallucination beat  *[INCLUDED — decided]*
Hand a plain LLM (or Leanstral without the verifier) a **subtly-false** variant of the spec →
it "proves" it convincingly → **Lean refuses**. ~60s; dramatizes points 1 & 3; preempts the skeptic.

## 🛠️ Setup status
- [x] Lean 4 toolchain present: elan 4.1.2, Lean/lake **v4.20.0**; `uv`/`uvx` present; `lms` CLI present.
- [x] LMStudio server up on **:1234** (other models loaded).
- [x] Fresh lake project scaffolded: `leanstral_demo/` (no mathlib — `omega` + core suffice).
- [x] **Repeatable layout** built + tested (see `leanstral_demo/DEMO_README.md`):
  - `problems/` pristine templates: `B_RevComp.lean`, `A_Interval.lean`, `B_RevComp_FALSE.lean`, `A_Interval_FALSE.lean`.
  - `runs/` throwaway working copies (gitignored); `solutions/` to bank verified proofs.
  - `scripts/new-run.sh {B|A|Bfalse|Afalse}` → fresh working copy; `scripts/check.sh <file>` → PASS/INCOMPLETE/FAIL; `scripts/reset.sh`.
- [x] **B** template: soft-mask + IUPAC. Base = symbol + `masked` flag; buggy table drops R/Y→N *and* resets mask; fixed `comp` preserves both. Machine-checked bug demos (`⟨A,true⟩`→`⟨A,false⟩`; `R`→`N`). Goal `revcomp_involution` = `sorry`.
- [x] **A** template: buggy `overlapBuggy` (`<=`), fixed `overlap` (`<`); machine-checked counterexample ([10,20) vs [20,30)). Goal `overlap_correct` = `sorry`.
- [x] **FALSE** templates + anti-hallucination path **validated**: pasting a bogus proof of a false claim → `check.sh` reports **FAIL** with the residual (false) goal; each file also carries a machine-checked disproof.
- [x] **A template corrected**: `overlap_correct` was false as first written (empty interval `lo ≥ hi` breaks the iff). Added non-emptiness preconditions `a.lo < a.hi`, `b.lo < b.hi` — and this "formalizing forced a hidden precondition" is now an email beat. FALSE variant brought into parallel.
- [x] **Banked fallback solutions** (both PASS, core Lean only, no mathlib): `solutions/B_RevComp.solved.lean`, `solutions/A_Interval.solved.lean`. B needs `List.map_map/map_reverse/reverse_reverse/map_id` + `funext`; A needs `omega` (Int max) + `decide_eq_true_eq`.
- [x] **Proving-loop prompt drafted**: `leanstral_demo/PROMPTS.md` (system + turn-0 + feedback turns, LMStudio settings, anti-hallucination variant).
- [ ] Reinstall **Leanstral 1.5** in LMStudio (~70GB, downloading) — then confirm served id on :1234.
- [ ] Wire Leanstral endpoint into the proving loop (hand it the two `sorry` goals).

## ✅ Results so far (2026-07-11) — BOTH TARGETS AUTONOMOUS
- **Target A SOLVED autonomously by Leanstral** — kernel-verified PASS in **3 rounds** (self-corrected: r1 `...` placeholder → r2 var-name slip → r3 PASS). Model wrote a real proof: witness `max a.lo b.lo`, `max_lt`/`le_max`, backward via `lt_of_le_of_lt`. Banked: `solutions/A_Interval.leanstral.lean`.
- **Target B SOLVED autonomously by Leanstral — via fair Pass@8.** Sequential single-thread feedback (temp 0.3) FAILED: it froze on one near-miss for 10 rounds (a ~pass@1 under-test). Running the benchmark's own regime — **8 independent samples, temp 0.8, ≤4 rounds each, no hint** (`scripts/best_of_n.sh B 8 4 0.8`) — cracked it on **attempt 6, round 1**, kernel-verified, no human edits. Banked: `solutions/B_RevComp.leanstral.lean`. (Interim artifacts kept: `B_RevComp.model_scaffolded.lean` = earlier human-repaired experiment; `B_RevComp.solved.lean` = human fallback.) **Lesson: measure autonomy in the model's intended regime (Pass@N + temp) before declaring a limit.**
- **Anti-hallucination beat DONE** — handed the FALSE claim (buggy `<=` overlap "is correct") to a plain local LLM (`qwen3.6-27b`); it confidently emitted a bogus proof; Lean **FAILed** it, while the same file machine-checks the disproof. Banked: `solutions/A_Interval_FALSE.plainllm_FAIL.lean` (+ `_reply.txt`).
- **Env correction:** **mathlib IS installed** in `leanstral_demo/` (the model's native env; templates `import Mathlib`) — supersedes the "no mathlib / core Lean only" notes in the older setup-status section above.
- **Harness lessons (most early "failures" were these, not the model):** mathlib required; LMStudio parallel→1 + 32k context stops MLX crashes; the model rambles across ~22 code blocks so extract PROOF-only via sentinels and splice into the pristine template; preserve/normalize proof indentation (dedent, don't lstrip line 1) or bullets misalign. Driver: `scripts/prove.py` (sentinel splice, crash-resilient retry, optional temperature arg); best-of-N: `scripts/best_of_n.sh`.

## ❓ Open questions
- LMStudio endpoint URL/port + served model id once reinstalled?
- Confirm 4-bit quant handles the multi-turn Lean-feedback loop acceptably, or do we drive turns manually?
- Video: rehearse-then-record, or one-and-done writeup first?
