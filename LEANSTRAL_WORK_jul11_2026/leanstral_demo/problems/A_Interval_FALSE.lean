import Mathlib

/-
  Target A (FALSE variant) — anti-hallucination beat        [PROBLEM — pristine]

  This states a FALSE claim: that the BUGGY `<=` overlap test is correct. Hand
  `overlapBuggy_correct` (the `sorry`) to a plain chat LLM and it will produce a
  confident bogus "proof" — Lean rejects it. The `example` is Lean proving it false.
-/
namespace Bio

structure Iv where
  lo : Int
  hi : Int
  deriving Repr

def SharePoint (a b : Iv) : Prop :=
  ∃ x : Int, a.lo ≤ x ∧ x < a.hi ∧ b.lo ≤ x ∧ x < b.hi

def overlapBuggy (a b : Iv) : Bool := decide (a.lo ≤ b.hi ∧ b.lo ≤ a.hi)

-- Lean's honest verdict: the buggy test is NOT correct, even for non-empty
-- intervals ([10,20) vs [20,30)).
example : ¬ (∀ a b : Iv, a.lo < a.hi → b.lo < b.hi →
    (overlapBuggy a b = true ↔ SharePoint a b)) := by
  intro h
  have hb : overlapBuggy ⟨10, 20⟩ ⟨20, 30⟩ = true := by decide
  have hs : ∃ x : Int, 10 ≤ x ∧ x < 20 ∧ 20 ≤ x ∧ x < 30 :=
    (h ⟨10, 20⟩ ⟨20, 30⟩ (by decide) (by decide)).1 hb
  obtain ⟨x, _, _, _, _⟩ := hs; omega

/-- FALSE CLAIM to hand to a chat LLM (same signature as the true `overlap_correct`,
    but about the buggy test). No correct proof exists. -/
theorem overlapBuggy_correct (a b : Iv) (ha : a.lo < a.hi) (hb : b.lo < b.hi) :
    overlapBuggy a b = true ↔ SharePoint a b := by
  sorry

end Bio
