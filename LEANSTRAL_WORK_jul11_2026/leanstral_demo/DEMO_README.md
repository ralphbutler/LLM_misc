# 🧪 Leanstral Demo — Workflow

Repeatable structure for running the formal-verification demos (testing *and* live/recorded).

## 📁 Layout
- `problems/` — **pristine templates** (source of truth). Never edited during a run.
  - `B_RevComp.lean` — revcomp involution (soft-mask + IUPAC). Goal: `revcomp_involution`.
  - `A_Interval.lean` — half-open interval overlap. Goal: `overlap_correct`.
  - `B_RevComp_FALSE.lean` / `A_Interval_FALSE.lean` — anti-hallucination beat (false claims).
- `runs/` — **throwaway working copies**, one per attempt (gitignored). We edit these.
- `solutions/` — **banked verified proofs** (fallbacks; copy a PASS-ing run here).
- `scripts/` — helpers below.

## 🔁 One attempt (test or demo)
```bash
f=$(scripts/new-run.sh B)     # fresh working copy -> prints its path, e.g. runs/B_20260711-094500.lean
# ... Leanstral fills in the proof, replacing `sorry` in $f ...
scripts/check.sh "$f"         # prints compiler output + VERDICT: PASS / INCOMPLETE / FAIL
```
Targets: `B`, `A`, `Bfalse`, `Afalse`.

Verdicts: **PASS** = kernel-verified, no `sorry`. **INCOMPLETE** = `sorry` remains.
**FAIL** = compiler rejected it (what happens when a bogus proof of a false claim is pasted in).

## 🏦 Banking a win
```bash
cp "$f" solutions/B_RevComp.solved.lean     # keep a verified fallback
```

## 🧹 Reset between takes
```bash
scripts/reset.sh              # clears runs/; templates untouched
```

## 🎬 Demo order
1. **B** — warmup, fast clean PASS (`revcomp_involution`).
2. **Anti-hallucination** — hand `Bfalse`/`Afalse` to a plain chat LLM, paste its "proof", get FAIL;
   contrast with the machine-checked disproof already in the file.
3. **A** — the real arc: show the counterexample, then PASS `overlap_correct` (∀ inputs). Gets the long budget.
