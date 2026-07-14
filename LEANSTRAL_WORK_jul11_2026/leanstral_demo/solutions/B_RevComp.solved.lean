import Mathlib

/-
  Target B — SOLVED FALLBACK (kernel-verified, no sorry).
  Keep as a safety net; the live goal for Leanstral is `revcomp_involution`.
-/
namespace Bio

inductive Sym | A | C | G | T | R | Y | N
  deriving DecidableEq, Repr

structure Base where
  sym    : Sym
  masked : Bool
  deriving DecidableEq, Repr

open Sym

def symComp : Sym → Sym
  | A => T | T => A | C => G | G => C | R => Y | Y => R | N => N

def symCompBuggy : Sym → Sym
  | A => T | T => A | C => G | G => C | R => N | Y => N | N => N

def comp (b : Base) : Base := { sym := symComp b.sym, masked := b.masked }
def compBuggy (b : Base) : Base := { sym := symCompBuggy b.sym, masked := false }

def revcomp (c : Base → Base) (s : List Base) : List Base := (s.map c).reverse

example : revcomp compBuggy (revcomp compBuggy [⟨A, true⟩]) = [⟨A, false⟩] := by decide
example : revcomp compBuggy (revcomp compBuggy [⟨A, true⟩]) ≠ [⟨A, true⟩]  := by decide
example : revcomp compBuggy (revcomp compBuggy [⟨R, false⟩]) = [⟨N, false⟩] := by decide
example : revcomp compBuggy (revcomp compBuggy [⟨R, false⟩]) ≠ [⟨R, false⟩] := by decide

theorem revcomp_involution (s : List Base) : revcomp comp (revcomp comp s) = s := by
  -- the fixed complement is its own inverse on every base
  have hc : ∀ b : Base, comp (comp b) = b := by
    intro b; cases b with | mk sym masked => cases sym <;> rfl
  unfold revcomp
  simp only [List.map_reverse, List.reverse_reverse, List.map_map]
  rw [show comp ∘ comp = id from funext hc, List.map_id]

end Bio
