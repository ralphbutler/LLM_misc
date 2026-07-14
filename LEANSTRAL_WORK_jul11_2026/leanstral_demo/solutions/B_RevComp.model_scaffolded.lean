/-
  Target B — revcomp involution.  PROVENANCE: MODEL-SCAFFOLDED, HUMAN-REPAIRED.
  This is NOT an autonomous Leanstral solve. Unaided (no hint, 10 rounds) Leanstral
  FAILED to produce a compiling proof — it converged by round 2 on an identical proof
  with several Lean-ergonomic defects and never escaped (see runs/B_unaided.log,
  runs/B_2026*.r*.reply.txt).

  What is Leanstral's (verbatim from its unaided attempts):
    - the entire STRATEGY (collapse revcomp∘revcomp via map/reverse identities),
    - all three helper-lemma STATEMENTS (h_comp_inv, h_map_rev, h_map_comp),
    - the observation `comp ∘ comp = id` (it explicitly raised `funext`, then discarded it).
  What a human fixed (Lean ergonomics only, NO new mathematics):
    - h_comp_inv: case on the Sym field (its `cases b` left symComp unreduced),
    - h_map_rev / h_map_comp: replace looping `simp [ih]` with curated `simp only`,
    - the calc: drop needless induction, fix `|>.reverse` syntax, use `← h_map_rev`,
      `List.reverse_reverse`, and the funext collapse it had discarded.
  Finding: a reasoning-vs-proof-assistant-fluency gap, not a math gap.
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
  -- The three helper lemmas below are exactly the ones Leanstral derived unaided.
  have h_comp_inv : ∀ b, comp (comp b) = b := by
    rintro ⟨s, m⟩; cases s <;> rfl
  have h_map_rev : ∀ (f : Base → Base) (l : List Base), (List.map f l).reverse = (l.reverse).map f := by
    intro f l; induction l with
    | nil => rfl
    | cons x xs ih => simp only [List.map_cons, List.reverse_cons, List.map_append, List.map_nil, ih]
  have h_map_comp : ∀ (f g : Base → Base) (l : List Base), List.map f (List.map g l) = List.map (f ∘ g) l := by
    intro f g l; induction l with
    | nil => rfl
    | cons x xs ih => simp only [List.map_cons, Function.comp_apply, ih]
  have hid : comp ∘ comp = id := funext (fun x => h_comp_inv x)
  -- Its strategy, assembled correctly (no induction on s needed; h_map_rev used the
  -- right direction; comp ∘ comp collapsed via funext before the map).
  calc
    revcomp comp (revcomp comp s)
        = (((s.map comp).reverse).map comp).reverse := rfl
    _ = (((s.map comp).map comp).reverse).reverse := by rw [← h_map_rev]
    _ = (s.map comp).map comp := by rw [List.reverse_reverse]
    _ = List.map (comp ∘ comp) s := by rw [h_map_comp]
    _ = s := by rw [hid, List.map_id]


end Bio
