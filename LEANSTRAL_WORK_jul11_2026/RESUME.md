# 🧭 RESUME — read this first

Living handoff so a new session (or a rested one) can continue seamlessly.
**Working dir:** `/Users/rbutler/Desktop/DEMO/LEANSTRAL`  ·  **User:** rbutler@mtsu.edu
**Last updated:** 2026-07-11, paused for ~1h token-limit reset.

## 🎯 What this project is
Use **Leanstral 1.5** (Mistral's Lean 4 prover LLM, running locally as a 4-bit MLX
quant in LMStudio) to do a *meaningful* formal-verification solve, then write it up for
RB's colleagues. Deliverable = an **email** (maybe a rehearsed video later). Bar: must be
genuinely meaningful, not "look, it does something."

## 🏆 HEADLINE RESULT (achieved) — BOTH TARGETS AUTONOMOUS
**Leanstral autonomously proved BOTH meaningful targets, kernel-verified, zero hints:**
- **A** (interval overlap) — PASS in **3 rounds**, self-correcting from compiler feedback (r1 `...`
  placeholder, r2 var-name slip, r3 clean). Real proof (witness `max a.lo b.lo`, `max_lt`/`le_max`,
  backward via `lt_of_le_of_lt`). Banked: `solutions/A_Interval.leanstral.lean`.
- **B** (revcomp involution) — PASS via **fair Pass@8** (8 independent samples, temp 0.8, no hint),
  solved on attempt 6, round 1. Banked: `solutions/B_RevComp.leanstral.lean`.
Plus the **anti-hallucination beat**: a plain local LLM confidently "proves" the FALSE claim → Lean FAILs it.

A's arc gives the email its centerpiece — the full honest arc on a meaningful target:
1. **Bug shown**: buggy `<=` overlap calls adjacent features `[10,20)`/`[20,30)` overlapping (machine-checked).
2. **Fix**: strict `<`.
3. **Autonomously proven correct for all valid intervals**, certified by the Lean kernel — zero trust in the LLM.
Bonus beat: formalizing *forced out* the hidden non-emptiness precondition.

Honest framing for colleagues: the 4-bit model **is** capable. Nearly every earlier
"failure" was OUR harness (see gotchas), not its reasoning. **UPDATE 2026-07-11: B is now
also an AUTONOMOUS solve** — a fair local **Pass@8** (8 independent samples, temp 0.8, no hint)
cracked it on attempt 6, round 1. So BOTH targets are autonomous. The earlier B "limit" was a
sampling-diversity shortfall (we'd run ~pass@1 at temp 0.3), exactly the paper's test-time-scaling
story — NOT a reasoning limit. See NEXT MOVES #2.

## 🧑‍🤝‍🧑 Working mode
- RB + Claude do a real solve together, then write it up. Claude leads and runs the Lean
  toolchain + model calls; RB narrates/presents. RB is new-ish to Lean 4 but fluent in
  classical ATP (E-prover/TPTP). Audience: mixed comp-bio + CS lab; none are mathematicians.
- RB prefs: be concise; offer better ideas proactively; emojis on main .md headings; end
  plans with a terse unresolved-questions list; RB often runs tests himself.

## ⚙️ Environment / model serving
- **Model** served on `http://localhost:1234`, id **`leanstral-1.5-119b-a6b-mlx`** (68GB, MLX).
  Load it with **parallel 1** and **context 32768** (this stopped the MLX crashes). Check with `lms ps`.
- Lean 4 **v4.20.0** (elan/lake); `uv`/`uvx`/`lms` present.
- **Mathlib IS installed** in `leanstral_demo/` (added via `lake update`; prebuilt cache auto-fetched).
  Each `lake env lean` check loads mathlib (~10-15s) — normal.

## ▶️ HOW TO RUN A PROVING ATTEMPT
```bash
cd leanstral_demo
lms ps                                  # confirm model loaded, parallel 1, ctx 32768
f=$(scripts/new-run.sh A)               # targets: A | B | Afalse | Bfalse  -> prints runs/<...>.lean
python3 scripts/prove.py "$f" 8 2>&1 | tee runs/A.log   # runs the loop (background it for long runs)
scripts/check.sh "$f"                   # PASS / INCOMPLETE / FAIL
```
Run long loops as a background shell + a Monitor on the log for per-round verdicts.
The driver extracts the model's PROOF between `###PROOF_BEGIN###/###PROOF_END###`, splices
it into the pristine template, compiles, and feeds errors back (crash-resilient retry built in).

## 🎯 NEXT MOVES (pick up here)
1. **✅ DONE (2026-07-11): Anti-hallucination beat for A.** Co-loaded a plain general LLM
   (`qwen3.6-27b-mlx` as identifier **`plainllm`**, ~35GB, alongside Leanstral — both fit in
   128GB) and handed it the false `overlapBuggy_correct`. It confidently emitted a bogus proof
   (7.3k reasoning tokens, self-assured); Lean **FAILed** it (`simp_all made no progress`).
   Contrast banked: `solutions/A_Interval_FALSE.plainllm_FAIL.lean` (re-verifiable, still FAIL) +
   `..._reply.txt` (raw). Written up in `REPORT_A.md` → "🚫 anti-hallucination beat" section.
   *(Optional extra: repeat for `Bfalse` if we want a second instance — not required.)*
   **To free memory before a big B run:** `lms unload plainllm`.
2. **✅ DONE (2026-07-11): B unaided, no hint — clean fair run at last.** 10 rounds, generic
   system prompt, temp 0.3, no math hint. **Result: FAIL (not solved).** But the *nature* of the
   failure is the story and it REFUTES the old "can't find the funext idiom" framing:
   - Unaided, it reconstructs the **entire correct strategy** and all three helper-lemma
     STATEMENTS (`h_comp_inv`, `h_map_rev`, `h_map_comp`); it even found a *funext-free* route
     (`map (comp∘comp)` → `map id`) and explicitly raised `funext` in round-1 reasoning.
   - It **converged by round 2** on one proof with several Lean-ergonomic defects (a `|>.reverse`
     calc-syntax slip, `cases b` not splitting the `Sym` field, looping `simp [ih]`) and **never
     escaped** — rounds 2→10 were byte-identical (r10 even regressed). **⇒ 6 rounds is plenty;
     more feedback rounds on one thread buy nothing here.** (Real diversity would need independent
     best-of-N at higher temp — untried, optional.)
   - **Post-hoc (legit, NOT a model assist):** taking the model's OWN lemmas + strategy and doing
     Lean-ergonomic repair ONLY (no new math) yields a kernel-verified proof:
     `solutions/B_RevComp.model_scaffolded.lean` (**PASS**, provenance header documents every edit).
     Honest label: *model-scaffolded, human-repaired* — NEVER "Leanstral proved B".
   - **Bottom line finding:** reasoning-vs-proof-assistant-fluency gap, not a math gap. This is the
     honest B counterpoint for the writeup.
2b. **✅ DONE (2026-07-11): B AUTONOMOUS via fair Pass@8.** After the sequential-feedback FAIL, we
   ran the benchmark's OWN method: `scripts/best_of_n.sh B 8 4 0.8` (8 independent fresh attempts,
   temp 0.8, ≤4 rounds each, stop on first PASS — NOT a hint, it's test-time scaling via sampling
   diversity). **SOLVED on attempt 6, round 1** — a single sample, zero feedback, kernel-verified.
   Its proof is *cleaner* than the human-repaired one (it derived the Sym-field casing itself).
   Banked: `solutions/B_RevComp.leanstral.lean` (**PASS**, provenance header). Log: `runs/B_bestof8.log`.
   Lesson: measure autonomy with the model's intended regime (Pass@N + temp) before declaring a limit.
3. **▶ NEXT: Write the email** — full, strong, honest story now: **A autonomous PASS** +
   **anti-hallucination beat** (plain LLM FAIL) + **B autonomous PASS via Pass@8**. Both meaningful
   targets solved by the local 4-bit model with zero hints; kernel-certified; the false-claim beat
   shows why the kernel (not the LLM) is the authority.

## 🗂️ File map
- `RESUME.md` (this) · `PLAN01.md` (decisions + "Results so far") · the release-blog PDF · `CONV01.txt` (raw convo export).
- `leanstral_demo/` — Lean project + demo kit:
  - `DEMO_README.md` (workflow), `PROMPTS.md` (prompt reference).
  - `problems/` pristine templates (each ends `:= by  sorry`; every bug-demo/disproof is machine-checked):
    `A_Interval.lean` (goal `overlap_correct`), `B_RevComp.lean` (goal `revcomp_involution`),
    `A_Interval_FALSE.lean`, `B_RevComp_FALSE.lean`. **All now start with `import Mathlib`.**
  - `solutions/` verified proofs: `A_Interval.leanstral.lean` (**Leanstral-authored**),
    `A_Interval.solved.lean` + `B_RevComp.solved.lean` (human fallbacks).
  - `runs/` throwaway working copies + `.rN.reply.txt` raw model replies (gitignored).
  - `scripts/`: `prove.py` (the driver), `new-run.sh`, `check.sh`, `reset.sh`,
    `test_candidates.py` / `test_candidates_A.py` (pre-test which proofs compile), `test_splice.py`, `analyze_reply.py`.

## 🧪 Targets
- **A — interval overlap** (`overlap_correct`, needs non-empty preconditions): **SOLVED by Leanstral**.
- **B — revcomp involution** (`revcomp_involution`, soft-mask + IUPAC): **AUTONOMOUS SOLVE via
  Pass@8** (attempt 6, round 1; temp 0.8; no hint; 2026-07-11). Banked: `B_RevComp.leanstral.lean`
  (autonomous, PASS). Also on file: `B_RevComp.solved.lean` (human) + `B_RevComp.model_scaffolded.lean`
  (the earlier repair experiment). Sequential-feedback run FAILED (temp 0.3, ~pass@1): `runs/B_unaided.log`.
  Pass@8 run: `runs/B_bestof8.log`, `runs/B_bo*.log`.

## ⚠️ Gotchas / lessons (do NOT rediscover)
- **Mathlib required** — it's the model's native env; without it the model reaches for
  mathlib idioms (`induction'`, mathlib lemmas) and flails. Templates now `import Mathlib`.
- **MLX crashes** on long generations were fixed by **parallel 1** (was 4 → ~4× KV cache) and
  ctx 32768. Memory was never the issue (41% free). Longer context does NOT help resilience.
- **Model rambles across ~22 code blocks** — never trust "return the whole file." The driver
  asks for PROOF-ONLY between sentinels and splices into the fixed template.
- **Indentation**: preserve the model's relative indentation, `textwrap.dedent`, then apply ONE
  2-space base indent. Do NOT `.strip()` the first line (misaligns `·` bullets → parse errors).
- Every template bug-demo/disproof is machine-checked, so a fresh template = INCOMPLETE (only the goal `sorry`).
- Pre-test candidate proofs locally with `test_candidates*.py` before blaming the model.

## ❓ Open questions
- Which next move first — anti-hallucination beat, B retry, or start the email?
- Video: rehearse-then-record, or email-only first?
- For B: (resolved) try UNAIDED first, no math hint — see NEXT MOVES #2. Only if it genuinely
  can't after a fair run do we discuss framing; never quietly fall back to a hinted proof.
