/-
  Target B — revcomp involution.  PROVENANCE: AUTONOMOUS LEANSTRAL SOLVE (kernel-verified).
  Method: fair local Pass@8 — 8 INDEPENDENT fresh attempts, temperature 0.8, <=4 feedback
  rounds each, no hint, no human edits to the proof. Solved on ATTEMPT 6, ROUND 1 (a single
  sample, zero feedback needed). Driver: scripts/best_of_n.sh B 8 4 0.8  (log: runs/B_bestof8.log).

  This REVERSES the earlier unaided FAIL (10 sequential feedback rounds, temp 0.3, which locked
  onto one near-miss). The difference was SAMPLING DIVERSITY, not a hint — exactly the paper's
  test-time-scaling story. The model even derived, on its own, the Sym-field casing that a human
  had to add by hand in the earlier B_RevComp.model_scaffolded.lean.
  Proof below is spliced verbatim from the model's reply; only the pristine template around it.
-/
import Mathlib

/-
  Target B — Reverse-complement round-trip (involution)     [PROBLEM — pristine]

  A base = a nucleotide symbol + a soft-mask flag. Soft-masking is the lowercase
  convention in FASTA (e.g. `a` vs `A`) that marks repetitive regions; it must
  survive reverse-complementing. Two real, common bugs live in one table:

    (1) IUPAC codes dropped: R (purine A/G) should complement to Y (pyrimidine
        C/T), but an uppercase-ACGT-only table sends R -> N.
    (2) Soft-mask lost: the complement resets the mask, so lowercase `a`
        round-trips to uppercase `A`.

  Either bug breaks the invariant revcomp(revcomp(s)) = s that tools assume.

  GOAL FOR THE PROVER: close `revcomp_involution` (currently `sorry`).
-/
namespace Bio

/-- Nucleotide symbol, including IUPAC ambiguity codes R (A/G), Y (C/T), and N. -/
inductive Sym | A | C | G | T | R | Y | N
  deriving DecidableEq, Repr

/-- A base: its symbol plus a soft-mask flag (`masked = true` ↔ lowercase in FASTA). -/
structure Base where
  sym    : Sym
  masked : Bool
  deriving DecidableEq, Repr

open Sym

/-- Correct symbol complement: R ↔ Y, and it is its own inverse. -/
def symComp : Sym → Sym
  | A => T | T => A | C => G | G => C | R => Y | Y => R | N => N

/-- BUGGY symbol complement: uppercase-ACGT table, IUPAC codes fall through to N. -/
def symCompBuggy : Sym → Sym
  | A => T | T => A | C => G | G => C | R => N | Y => N | N => N

/-- Correct base complement: complement the symbol, PRESERVE the soft-mask flag. -/
def comp (b : Base) : Base := { sym := symComp b.sym, masked := b.masked }

/-- BUGGY base complement: drops IUPAC codes (via `symCompBuggy`) AND resets the
    soft-mask to false — so lowercase bases silently become uppercase. -/
def compBuggy (b : Base) : Base := { sym := symCompBuggy b.sym, masked := false }

/-- Reverse-complement, parameterized by the complement function. -/
def revcomp (c : Base → Base) (s : List Base) : List Base := (s.map c).reverse

-- The bugs, made concrete (all machine-checked):
-- (2) soft-mask bug: round-tripping lowercase `a` = ⟨A,true⟩ loses the mask.
example : revcomp compBuggy (revcomp compBuggy [⟨A, true⟩]) = [⟨A, false⟩] := by decide
example : revcomp compBuggy (revcomp compBuggy [⟨A, true⟩]) ≠ [⟨A, true⟩]  := by decide
-- (1) IUPAC bug: round-tripping R = ⟨R,false⟩ corrupts it to N.
example : revcomp compBuggy (revcomp compBuggy [⟨R, false⟩]) = [⟨N, false⟩] := by decide
example : revcomp compBuggy (revcomp compBuggy [⟨R, false⟩]) ≠ [⟨R, false⟩] := by decide

/-- SPEC (the prover's job): with the fixed complement, reverse-complement is an
    involution for EVERY sequence, soft-masked and IUPAC bases included. -/
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


end Bio
