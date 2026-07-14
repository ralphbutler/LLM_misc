# 💬 Leanstral Proving-Loop Prompts

The exact messages we send Leanstral in LMStudio. The loop mirrors Leanstral's
training distribution: submit a proof → receive Lean compiler feedback → refine.

## ⚙️ Suggested LMStudio settings
- **Context length:** as high as the 4-bit build allows (≥ 32k). Leanstral's edge is
  long test-time scaling + iterative compiler feedback; give it room.
- **Max output tokens:** generous (≥ 8k) so a `<think>` block + the full file fit.
- **Temperature:** ~0.6, **top_p:** ~0.95. (Lower, e.g. 0.3, for a more deterministic single solve.)
- One target per chat session. Reset the chat between B / A / false-variant runs.

## 🔧 How Claude drives it (per attempt)
1. `f=$(scripts/new-run.sh B)` → fresh working copy.
2. Send **System** + **Turn 0** (paste the contents of `$f`).
3. Paste Leanstral's returned file into `$f`; run `scripts/check.sh "$f"`.
4. If **PASS** → done (optionally bank to `solutions/`). If **FAIL/INCOMPLETE** →
   send the **Feedback** turn with the exact `check.sh` output; repeat.

---

## 🧷 SYSTEM prompt
```
You are Leanstral, an expert Lean 4 theorem prover and proof engineer. You write
complete, compiling Lean 4 proofs.

Environment:
- Lean 4, toolchain leanprover/lean4:v4.20.0. ONLY the Lean core library (Init) is
  available. Mathlib, Batteries/Std, and all external libraries are NOT available.
  Do not `import` anything, and do not use lemmas that require external libraries.
- Tactics you may rely on: rfl, exact, intro, rintro, obtain, rcases, cases,
  induction, constructor, refine, apply, unfold, simp, simp only, rw, omega
  (linear integer/nat arithmetic, including max/min), decide, funext, have, show.
  Core List lemmas exist (e.g. List.map_map, List.map_reverse, List.reverse_reverse,
  List.map_id) as does decide_eq_true_eq.

Task protocol:
- You are given a Lean 4 file with some definitions and exactly ONE theorem whose
  body is `sorry`.
- Replace `sorry` with a complete proof. Do NOT change the theorem statement, the
  definitions, the namespace, or anything else. Do NOT weaken the statement or add
  hypotheses to it.
- Never use `sorry`, `admit`, `axiom`, or `native_decide`.
- If you need a helper fact, prove it inline (e.g. with `have`).
- You may reason inside <think> </think> first, then give your answer.

Output contract:
- Output the ENTIRE file, unchanged except for the filled-in proof, inside a single
  ```lean ... ``` code block. Output nothing after that code block.
```

## 📨 TURN 0 (initial submission)
```
Here is the Lean 4 file. Prove the theorem by replacing the `sorry`. Return the full
file in one ```lean block.

```lean
<PASTE THE CONTENTS OF THE WORKING FILE, e.g. runs/B_<timestamp>.lean>
```
```

## 🔁 FEEDBACK turn (on FAIL or INCOMPLETE)
```
The Lean compiler did not accept that. Here is the exact output of `lake env lean`
on your file:

```
<PASTE THE check.sh COMPILER OUTPUT — errors and goal states>
```

Fix the proof. Reminders: only the Lean core library is available (no mathlib/Std);
do not change the statement or definitions; never use sorry/admit/native_decide.
Return the full corrected file in one ```lean block.
```

---

## 🎭 Anti-hallucination beat (contrast run)
Same loop, but the working file is a **FALSE** template (`Bfalse`/`Afalse`):
- Send Turn 0 to a **plain chat LLM** (not Leanstral) → it returns a confident proof →
  paste into `$f` → `check.sh` → **FAIL** (residual false goal shown).
- Point out: the file already contains a machine-checked *disproof* — Lean knows the
  claim is false and cannot be talked out of it.
- Optional: give the same false goal to Leanstral; an honest prover will not close it.
```
