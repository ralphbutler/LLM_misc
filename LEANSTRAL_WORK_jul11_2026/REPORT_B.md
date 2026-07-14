# 🧬 Report B — Leanstral autonomously proves a reverse-complement round-trip correct

*Draft — 2026-07-11. Companion to REPORT_A. Target B (revcomp involution). The distinctive beat
here is the honest FAIL→PASS arc: sequential feedback stalled, and the model's own test-time-scaling
regime (independent sampling) cracked it — no hint.*

## 📌 TL;DR
Running **Leanstral 1.5** locally (a 4-bit quant on one workstation, in LM Studio), we had it prove
that a **fixed** reverse-complement function is a true *involution* — `revcomp(revcomp(s)) = s` for
**every** sequence, soft-masked and IUPAC bases included — in the **Lean 4** proof assistant. It
found a real, clean proof and the Lean kernel **certified** it. The honest twist: our first regime
(one thread, low temperature, up to 10 compiler-feedback rounds) **failed** — the model froze on a
single near-miss. Switching to the prover's *intended* regime — **8 independent samples at higher
temperature (a fair Pass@8)** — it solved the goal on the 6th sample, first try, with **no hint and
no human edits**. The lesson is as useful as the proof: *measure a prover's autonomy in the regime
it was built for before calling something a limit.*

## 🐞 The problem (two real reference-genome bugs in one table)
Reverse-complement is the most-run operation in genomics: to read the opposite DNA strand you
reverse the sequence and complement each base (A↔T, C↔G). Tools everywhere assume the round-trip is
an **involution** — do it twice, get the original back. Two common bugs quietly break that invariant:

1. **IUPAC codes dropped.** Ambiguity codes are real bases: `R` = purine (A/G), which must
   complement to `Y` = pyrimidine (C/T). An uppercase-ACGT-only table sends `R → N`, corrupting it.
2. **Soft-mask lost.** Lowercase in FASTA (`a` vs `A`) marks repetitive regions and must survive
   complementing. A buggy complement resets the flag, so lowercase `a` round-trips to uppercase `A`.

We encoded a base as *symbol + soft-mask flag* (`structure Base`), with the correct complement
`comp` (fixes both: `symComp` maps `R↔Y` and preserves the mask) and a buggy `compBuggy` (drops
IUPAC via `symCompBuggy`, and resets the mask to `false`). Both bugs are **machine-checked**, not
asserted — Lean itself confirms the round-trip corrupts the data:

- soft-mask bug: `revcomp compBuggy (revcomp compBuggy [⟨A, true⟩]) = [⟨A, false⟩]` (mask lost).
- IUPAC bug: `revcomp compBuggy (revcomp compBuggy [⟨R, false⟩]) = [⟨N, false⟩]` (base corrupted).

(`leanstral_demo/problems/B_RevComp.lean`.)

## 🔧 The fix and the claim to prove
- **Fixed complement:** `symComp` completes the IUPAC pairing (`R↔Y`) and `comp` **preserves** the
  soft-mask flag.
- **Specification (the goal):** with the fixed complement, reverse-complement is an **involution
  for every sequence** — `revcomp comp (revcomp comp s) = s` — soft-masked and IUPAC bases included.

The math is trivial to state; the challenge for a machine prover is *proof-assistant ergonomics* —
getting Lean to see that `comp ∘ comp = id` collapses under `List.map`, and that the two reverses
cancel.

## 🤖 What Leanstral did — the honest FAIL→PASS arc
We handed Leanstral the file with the theorem left as `sorry`. **Two regimes, one honest story:**

**Regime 1 — sequential feedback (FAILED).** One thread, temperature 0.3, up to 10 rounds of
compiler-error feedback. It reconstructed the *entire correct strategy* unaided — the right helper
lemmas (`comp ∘ comp = id`, map/reverse commutation) — but its proof had small Lean-ergonomic
defects, and **it converged by round 2 and never escaped**: rounds 2→10 were byte-identical. More
feedback on the same thread bought nothing. (Raw: `runs/B_unaided.log`.) At this point it looked
like a genuine limit — but that conclusion was premature, because we hadn't run the prover the way
it's designed to run.

**Regime 2 — fair Pass@8 (SOLVED).** Leanstral's published benchmark numbers come from *test-time
scaling*: many independent samples, generous budgets. Sequential feedback at temp 0.3 is effectively
**pass@1** with almost no diversity. So we ran the model's own regime — **8 independent fresh
attempts, temperature 0.8, ≤4 rounds each, stop at the first proof that compiles** (no hint; higher
temperature and independent draws are *sampling diversity*, not guidance). It solved the goal on
**attempt 6, round 1** — a single sample, zero feedback, kernel-verified. Notably, that sample
derived on its own the exact `Sym`-field case split that a human had needed to add by hand in an
earlier repair experiment. The proof it wrote:

```lean
theorem revcomp_involution (s : List Base) : revcomp comp (revcomp comp s) = s := by
  have h_symComp_invol : ∀ s, symComp (symComp s) = s := by
    intro s; cases s <;> rfl
  have h_comp_invol : ∀ b, comp (comp b) = b := by
    intro b; cases b <;> simp [comp, h_symComp_invol]
  have h_map_rev : ∀ s : List Base, (s.map comp).reverse.map comp = s.reverse := by
    intro s; induction s with
    | nil => rfl
    | cons b bs ih =>
      simp [List.map_cons, List.reverse_cons, h_comp_invol, ih]
  calc
    revcomp comp (revcomp comp s) = ((s.map comp).reverse.map comp).reverse := rfl
    _ = (s.reverse).reverse := by rw [h_map_rev]
    _ = s := List.reverse_reverse _
```

It is real, structured mathematics: prove `symComp` self-inverse by cases, lift it to `comp`
(preserving the mask), fold the map/reverse/complement collapse into one induction, then cancel the
two reverses. Banked and independently re-verified: `leanstral_demo/solutions/B_RevComp.leanstral.lean`.

## 🎯 Why this matters
Everything in Report A's "why it matters" applies — a **certificate, not a guess**; proven for
**∀ sequences**, not sampled ones; the kernel, not the model, is the authority. Target B adds one
more, methodological, point that is worth telling colleagues plainly:

- **Autonomy is a function of the regime.** The same model, on the same problem, went from a
  confident-looking *failure* to a clean *autonomous solve* purely by giving it its intended
  test-time budget (independent sampling). The failure wasn't reasoning; it was our under-powered
  harness. When you evaluate a local prover, replicate its regime (Pass@N, temperature, budget)
  before concluding it "can't."

## 🔬 Honesty about the setup
- **Local, 4-bit, one machine.** Same caveats as A. The full-precision Leanstral saturates these
  benchmarks with far larger budgets; our local quant is weaker but clearly capable of this proof.
- **Sequential feedback failed first; sampling diversity succeeded.** We report both. The Pass@8
  win took 6 independent samples (~21 total generations) — this is the prover's normal mode, not a
  lucky fluke, but it is more compute than A's tidy 3-round solve.
- **No hint, no human edits to the proof.** Higher temperature and independent attempts are
  test-time scaling, and the compiler-feedback loop is how the tool is designed to work — neither
  supplies mathematics. The proof above is spliced verbatim from the model's reply.
- **A note on an interim experiment.** Before the Pass@8 run we hand-repaired the model's Regime-1
  near-miss into a compiling proof (`solutions/B_RevComp.model_scaffolded.lean`). That was
  *human-completed* and we do **not** count it as an autonomous result; it is kept only to document
  where Regime 1 fell short. The autonomous result is the Pass@8 proof.

## 🧪 Reproduce it
```bash
cd leanstral_demo
lms ps                                   # Leanstral loaded (parallel 1, ctx 32768)
scripts/best_of_n.sh B 8 4 0.8           # 8 independent attempts, temp 0.8, stop on first PASS
#   -> "@@@@@ Pass@8 SUCCESS on attempt N: runs/B_<ts>.lean @@@@@"
scripts/check.sh solutions/B_RevComp.leanstral.lean   # re-verify the banked proof: VERDICT: PASS
```
(Because sampling is stochastic, the winning attempt number will vary run to run; the banked proof
in `solutions/` is the exact one from our run and re-verifies deterministically.)

## 🚧 TODO (enhance after we finish the demo)
- Tighten for the email: pair with Report A's centerpiece; B contributes the honest test-time-scaling
  beat (feedback stalled → sampling diversity solved it, no hint).
- Optional: report the empirical pass rate (attempts-to-solve) if we want a quantitative autonomy number.
