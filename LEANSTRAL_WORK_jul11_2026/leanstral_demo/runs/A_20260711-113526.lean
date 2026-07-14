import Mathlib

/-
  Target A — 0-based / 1-based interval overlap (off-by-one)  [PROBLEM — pristine]

  Genomic intervals are half-open [lo, hi) (the BED convention). The classic bug
  is testing overlap with `<=` (as if intervals were closed), which reports
  ADJACENT features as overlapping. An exon [10,20) and the next feature [20,30)
  touch but do not overlap — yet the buggy test says they do, so a variant at
  position 20 gets annotated to the wrong gene.

  GOAL FOR THE PROVER: close `overlap_correct` (currently `sorry`).
-/
namespace Bio

/-- Half-open genomic interval [lo, hi) (BED convention). -/
structure Iv where
  lo : Int
  hi : Int
  deriving Repr

/-- Semantic overlap: the two intervals share at least one integer coordinate.
    This is the ground truth an overlap test is supposed to compute. -/
def SharePoint (a b : Iv) : Prop :=
  ∃ x : Int, a.lo ≤ x ∧ x < a.hi ∧ b.lo ≤ x ∧ x < b.hi

/-- BUGGY test: `<=` on half-open coordinates (treats intervals as closed).
    Reports touching-but-disjoint features as overlapping. -/
def overlapBuggy (a b : Iv) : Bool := decide (a.lo ≤ b.hi ∧ b.lo ≤ a.hi)

/-- Correct test: strict `<` on half-open coordinates. -/
def overlap (a b : Iv) : Bool := decide (a.lo < b.hi ∧ b.lo < a.hi)

-- The bug, made concrete (all machine-checked):
-- adjacent intervals [10,20) and [20,30) share no point ...
example : ¬ SharePoint ⟨10, 20⟩ ⟨20, 30⟩ := by
  show ¬ ∃ x : Int, 10 ≤ x ∧ x < 20 ∧ 20 ≤ x ∧ x < 30
  rintro ⟨x, _, _, _, _⟩; omega
-- ... yet the buggy test calls them overlapping ...
example : overlapBuggy ⟨10, 20⟩ ⟨20, 30⟩ = true  := by decide
-- ... while the fixed test correctly says they don't.
example : overlap    ⟨10, 20⟩ ⟨20, 30⟩ = false := by decide

/-- SPEC (the prover's job): the fixed test agrees with ground-truth overlap for
    every pair of NON-EMPTY intervals. The non-emptiness preconditions are not
    decoration — without them the claim is false (an empty interval lo ≥ hi can
    satisfy the `<` test while sharing no point). Formalizing forces that hidden
    assumption into the open. -/
theorem overlap_correct (a b : Iv) (ha : a.lo < a.hi) (hb : b.lo < b.hi) :
    overlap a b = true ↔ SharePoint a b := by
  constructor
  · intro h
    have hdec : (a.lo < b.hi ∧ b.lo < a.hi) := by
      simpa [overlap] using h
    rcases hdec with ⟨hlt1, hlt2⟩
    set x := max a.lo b.lo with hxdef
    have hax : a.lo ≤ x := by
      simp [hxdef]
    have hbx : b.lo ≤ x := by
      simp [hxdef]
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


end Bio
