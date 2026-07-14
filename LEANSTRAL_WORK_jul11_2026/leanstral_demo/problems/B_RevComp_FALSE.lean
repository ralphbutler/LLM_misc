import Mathlib

/-
  Target B (FALSE variant) — anti-hallucination beat        [PROBLEM — pristine]

  This states a FALSE claim: that the BUGGY reverse-complement is an involution.
  Hand `revcomp_buggy_involution` (the `sorry`) to a plain chat LLM and it will
  happily emit a convincing-looking "proof" — paste it in and Lean REJECTS it,
  because the statement is false. The `example` below is Lean proving it false.
-/
namespace Bio

inductive Sym | A | C | G | T | R | Y | N
  deriving DecidableEq, Repr

structure Base where
  sym    : Sym
  masked : Bool
  deriving DecidableEq, Repr

open Sym

def symCompBuggy : Sym → Sym
  | A => T | T => A | C => G | G => C | R => N | Y => N | N => N

def compBuggy (b : Base) : Base := { sym := symCompBuggy b.sym, masked := false }

def revcomp (c : Base → Base) (s : List Base) : List Base := (s.map c).reverse

-- Lean's honest verdict: the claim is FALSE (lowercase `a` = ⟨A,true⟩ is a witness).
example : ¬ (∀ s : List Base, revcomp compBuggy (revcomp compBuggy s) = s) := by
  intro h; exact absurd (h [⟨A, true⟩]) (by decide)

/-- FALSE CLAIM to hand to a chat LLM. No correct proof exists; Lean cannot be
    fooled into closing this goal. -/
theorem revcomp_buggy_involution (s : List Base) :
    revcomp compBuggy (revcomp compBuggy s) = s := by
  sorry

end Bio
