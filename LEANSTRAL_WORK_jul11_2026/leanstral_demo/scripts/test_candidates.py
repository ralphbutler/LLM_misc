import importlib.util, pathlib, subprocess
spec = importlib.util.spec_from_file_location("p", "scripts/prove.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
template = pathlib.Path("problems/B_RevComp.lean").read_text()

candidates = {
  "A induction<;>simp_all": "induction s <;> simp_all [revcomp, comp, symComp]",
  "B simp map lemmas": "simp only [revcomp, List.map_reverse, List.reverse_reverse, List.map_map]\nsimp [comp, symComp, Function.comp]",
  "C hc + simp": ("have hc : ∀ b : Base, comp (comp b) = b := by\n"
                   "  intro b; cases b with | mk s m => cases s <;> rfl\n"
                   "simp only [revcomp, List.map_reverse, List.reverse_reverse, List.map_map, Function.comp, hc, List.map_id]"),
  "D just simp": "simp [revcomp, comp, symComp, Function.comp]",
}
for name, proof in candidates.items():
    f = pathlib.Path("runs/B_cand.lean")
    f.write_text(p.splice(template, proof))
    r = subprocess.run(["scripts/check.sh", "runs/B_cand.lean"], capture_output=True, text=True)
    verdict = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
    print(f"{name:28s} -> {verdict}")
pathlib.Path("runs/B_cand.lean").unlink(missing_ok=True)
