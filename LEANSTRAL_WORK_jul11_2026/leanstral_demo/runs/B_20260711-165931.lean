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
  induction s with
  | nil => rfl
  | cons b bs ih =>
    have h_sym : ∀ x : Sym, symComp (symComp x) = x := by
      intro x; cases x <;> rfl
    simp [revcomp, comp, h_sym, ih]


end Bio
