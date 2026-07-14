import Mathlib

/-
  Target A — SOLVED FALLBACK (kernel-verified, no sorry).
  Keep as a safety net; the live goal for Leanstral is `overlap_correct`.
-/
namespace Bio

structure Iv where
  lo : Int
  hi : Int
  deriving Repr

def SharePoint (a b : Iv) : Prop :=
  ∃ x : Int, a.lo ≤ x ∧ x < a.hi ∧ b.lo ≤ x ∧ x < b.hi

def overlapBuggy (a b : Iv) : Bool := decide (a.lo ≤ b.hi ∧ b.lo ≤ a.hi)
def overlap (a b : Iv) : Bool := decide (a.lo < b.hi ∧ b.lo < a.hi)

example : ¬ SharePoint ⟨10, 20⟩ ⟨20, 30⟩ := by
  show ¬ ∃ x : Int, 10 ≤ x ∧ x < 20 ∧ 20 ≤ x ∧ x < 30
  rintro ⟨x, _, _, _, _⟩; omega
example : overlapBuggy ⟨10, 20⟩ ⟨20, 30⟩ = true  := by decide
example : overlap    ⟨10, 20⟩ ⟨20, 30⟩ = false := by decide

theorem overlap_correct (a b : Iv) (ha : a.lo < a.hi) (hb : b.lo < b.hi) :
    overlap a b = true ↔ SharePoint a b := by
  unfold overlap SharePoint
  constructor
  · intro h
    simp only [decide_eq_true_eq] at h
    -- a witness in the intersection: the larger of the two left endpoints
    exact ⟨max a.lo b.lo, by omega, by omega, by omega, by omega⟩
  · rintro ⟨x, h1, h2, h3, h4⟩
    simp only [decide_eq_true_eq]
    omega

end Bio
