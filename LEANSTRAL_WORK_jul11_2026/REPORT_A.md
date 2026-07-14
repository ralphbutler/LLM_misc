# 🧬 Report A — Leanstral autonomously proves a genomic-interval overlap function correct

*Draft — 2026-07-11. Centerpiece report on Target A. Anti-hallucination section and final
polish still TODO (marked below).*

## 📌 TL;DR
Running **Leanstral 1.5** locally (a 4-bit quant on one workstation, in LM Studio), we had it
tackle a classic bioinformatics bug class — off-by-one in half-open genomic interval overlap —
in the **Lean 4** proof assistant. It found and closed a real correctness proof: given the
*fixed* overlap test, it **autonomously proved the test agrees with true point-sharing for
every valid interval pair**, and the Lean kernel *certified* that proof. It did this in **3
feedback rounds**, correcting its own compiler errors along the way. The result is not "an AI
thinks the code is right" — it's a machine-checked guarantee that requires trusting no LLM.

## 🐞 The problem (a bug every comp-bio coder has hit)
Genomic intervals are half-open, `[lo, hi)` (the BED convention). Whether two features overlap
is decided constantly (does this variant fall in this exon? do these peaks intersect?). The
classic bug is writing the test with `<=` as if intervals were *closed*:

- **Buggy:** `a.lo <= b.hi ∧ b.lo <= a.hi` — reports *adjacent* features as overlapping.
- **Ground truth:** the intervals share at least one integer coordinate.

Concretely, an exon `[10, 20)` and the next feature `[20, 30)` **touch but do not overlap** —
yet the buggy test says they do, so a variant at position 20 gets annotated to the wrong gene.

We encoded this in Lean (`leanstral_demo/problems/A_Interval.lean`). The bug is **machine-checked**,
not asserted: Lean itself confirms the buggy test returns `true` on `[10,20)`/`[20,30)` while no
shared point exists, and that the fixed test (strict `<`) returns `false`.

## 🔧 The fix and the claim to prove
- **Fixed test:** `a.lo < b.hi ∧ b.lo < a.hi`.
- **Specification (the hero goal):** for every pair of intervals, the fixed test equals true
  **iff** the intervals actually share a point.

A subtlety the formalization *forced into the open*: the claim is **false without a
non-emptiness precondition** — an empty/malformed interval (`lo ≥ hi`) can pass the `<` test
while sharing no point. Informal reasoning glosses over this; Lean does not. So the proven
theorem carries `a.lo < a.hi` and `b.lo < b.hi` as explicit hypotheses. *(Good talking point:
formalizing surfaced a hidden assumption.)*

## 🤖 What Leanstral did (autonomously, in 3 rounds)
We handed Leanstral the file with the theorem left as `sorry` and ran a simple loop: it proposes
a proof, the Lean compiler judges it, and any errors go back for another attempt.

- **Round 1** → it drafted a correct *strategy* but left a `...` placeholder — rejected.
- **Round 2** → complete proof, but a bound referenced an unbound name (`hx1`) — rejected.
- **Round 3** → **clean proof; kernel-verified PASS.**

The proof it wrote is real mathematics, not a one-liner. It:
- reduces the boolean test to the arithmetic conjunction,
- constructs an explicit witness point, `max a.lo b.lo`, for the "they share a point" direction,
- bounds that witness with `max_lt` / `le_max_left`/`le_max_right`,
- and handles the converse with `lt_of_le_of_lt`.

```lean
theorem overlap_correct (a b : Iv) (ha : a.lo < a.hi) (hb : b.lo < b.hi) :
    overlap a b = true ↔ SharePoint a b := by
  constructor
  · intro h
    have hdec : (a.lo < b.hi ∧ b.lo < a.hi) := by
      simpa [overlap] using h
    rcases hdec with ⟨hlt1, hlt2⟩
    set x := max a.lo b.lo with hxdef
    have hax : a.lo ≤ x := by simp [hxdef]
    have hbx : b.lo ≤ x := by simp [hxdef]
    have hxa : x < a.hi := by
      have : max a.lo b.lo < a.hi := max_lt ha hlt2
      simpa [hxdef] using this
    have hxb : x < b.hi := by
      have : max a.lo b.lo < b.hi := max_lt hlt1 hb
      simpa [hxdef] using this
    exact ⟨x, hax, hxa, hbx, hxb⟩
  · intro h
    rcases h with ⟨x, hax, hxa, hbx, hxb⟩
    have hlt1 : a.lo < b.hi := lt_of_le_of_lt hax hxb
    have hlt2 : b.lo < a.hi := lt_of_le_of_lt hbx hxa
    simp [overlap, hlt1, hlt2]
```

Banked and independently re-verified: `leanstral_demo/solutions/A_Interval.leanstral.lean`.

## 🎯 Why this matters (and why it isn't "an LLM caught a bug")
A modern chatbot can *eyeball* this off-by-one. That is not the point. The point is what a proof
assistant adds that no LLM can:
1. **Certificate, not a guess.** The Lean kernel (a small, trusted checker) certifies the proof.
   You trust the kernel, not the model — the same model that could have written the bug.
2. **∀ inputs, not sampled ones.** The *fix* is the hero step: it's proven correct for **all**
   valid interval pairs, not "looks right on the cases we tried." Testing/fuzzing/eyeballing all
   sample; the proof quantifies universally.
3. **It exposes hidden assumptions.** The non-emptiness precondition was invisible until we
   formalized.

This example is deliberately small enough that you can verify the tool isn't cheating — but the
*same* machinery scales to code no human can check by inspection.

## 🔬 Honesty about the setup
- **Local, 4-bit, one machine.** No cloud, no big model. The full-precision Leanstral (per
  Mistral's release) saturates math benchmarks; our local quant is weaker but clearly capable.
- **Collaboration loop, not one-shot.** Leanstral proposed; the Lean compiler judged; it revised.
  Two of three rounds were the model fixing its *own* mistakes with no human hint. Realistic
  proof engineering looks like this.
- **Environment matters.** It runs in Lean 4 + Mathlib (the library it was trained on). The
  proof uses standard Mathlib lemmas (`max_lt`, `le_max_left`, `lt_of_le_of_lt`).

## 🧪 Reproduce it
```bash
cd leanstral_demo
lms ps                                   # Leanstral loaded (parallel 1, ctx 32768)
f=$(scripts/new-run.sh A)                # fresh copy of the problem
python3 scripts/prove.py "$f" 8          # run the propose→check→refine loop
scripts/check.sh "$f"                    # expect: VERDICT: PASS
```

## 🚫 The anti-hallucination beat (why the kernel, not the model, is the authority)
To show what the proof assistant *buys* you, we ran the mirror-image experiment. We stated the
**false** claim — that the *buggy* `≤` test is correct — and handed it to a **plain general chat
LLM** running locally alongside Leanstral (a 27B general model, *not* a Lean specialist), with the
exact same Lean signature and a `sorry` to fill.

It never hesitated. It "thought" for ~7,300 reasoning tokens, talked itself into confidence
(*"It's concise and correct under standard assumptions… All good. [Done]"*), and emitted a
tidy-looking proof:
```lean
theorem overlapBuggy_correct (a b : Iv) (ha : a.lo < a.hi) (hb : b.lo < b.hi) :
    overlapBuggy a b = true ↔ SharePoint a b := by
  simp [overlapBuggy, SharePoint]
  constructor <;> intro h <;>
    (try { use max a.lo b.lo }) <;>
    simp_all [max_le_iff] <;>
    omega
```
It *looks* like the real proof of Target A. It is worthless — the claim is false, so no proof
exists. **Lean rejected it outright:**
```
error: simp_all made no progress
VERDICT: FAIL  (compiler rejected it)
```
And in the *same* file, Lean does the positive thing an LLM cannot: it **machine-checks the
disproof** — it proves the buggy test is genuinely wrong, exhibiting the counterexample
`[10,20)` vs `[20,30)` (they touch, share no point, yet the buggy test says "overlap").

That is the whole thesis in one contrast:
- Leanstral's proof of the **true** claim → kernel says **PASS**.
- A confident LLM's proof of the **false** claim → kernel says **FAIL**.

The kernel — a small, trusted checker — is the authority. The LLM (any LLM, the same kind that
could have written the bug) is only a *proposer*. You are never asked to trust it.

*Artifacts:* `solutions/A_Interval_FALSE.plainllm_FAIL.lean` (the rejected proof, re-verifiable
with `scripts/check.sh`) and `solutions/A_Interval_FALSE.plainllm_reply.txt` (the raw LLM reply +
its reasoning tail).

## 🚧 TODO (enhance after we finish the demo)
- Companion **Report B** (revcomp involution): now also an **autonomous solve** — via fair Pass@8
  (8 independent samples, temp 0.8, no hint; solved attempt 6, round 1). See `REPORT_B.md`. Its
  arc adds the *test-time-scaling* lesson: sequential feedback froze at ~pass@1; sampling diversity
  cracked it — no hint. (Earlier "can't do B" framing was an under-powered harness, not a model limit.)
- Tighten for the email: lift the TL;DR + "why it matters" into a one-screen version.
