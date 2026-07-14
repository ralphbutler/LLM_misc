import importlib.util, pathlib, subprocess
spec = importlib.util.spec_from_file_location("p", "scripts/prove.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
template = pathlib.Path("problems/A_Interval.lean").read_text()

candidates = {
  "A1 banked constructor+max": (
    "unfold overlap SharePoint\n"
    "constructor\n"
    "· intro h\n"
    "  simp only [decide_eq_true_eq] at h\n"
    "  exact ⟨max a.lo b.lo, by omega, by omega, by omega, by omega⟩\n"
    "· rintro ⟨x, h1, h2, h3, h4⟩\n"
    "  simp only [decide_eq_true_eq]\n"
    "  omega"),
  "A2 simp+anon constructor": (
    "simp only [overlap, SharePoint, decide_eq_true_eq]\n"
    "exact ⟨fun h => ⟨max a.lo b.lo, by omega, by omega, by omega, by omega⟩,\n"
    "       fun ⟨x, h1, h2, h3, h4⟩ => by omega⟩"),
  "A3 refine+omega": (
    "simp only [overlap, SharePoint, decide_eq_true_eq]\n"
    "constructor\n"
    "· rintro ⟨h1, h2⟩\n"
    "  refine ⟨max a.lo b.lo, ?_, ?_, ?_, ?_⟩ <;> omega\n"
    "· rintro ⟨x, h1, h2, h3, h4⟩\n"
    "  omega"),
  "A4 min witness": (
    "simp only [overlap, SharePoint, decide_eq_true_eq]\n"
    "constructor\n"
    "· rintro ⟨h1, h2⟩\n"
    "  exact ⟨max a.lo b.lo, le_max_left .. , by omega, le_max_right .., by omega⟩\n"
    "· rintro ⟨x, h1, h2, h3, h4⟩\n"
    "  omega"),
}
for name, proof in candidates.items():
    f = pathlib.Path("runs/A_cand.lean")
    f.write_text(p.splice(template, proof))
    r = subprocess.run(["scripts/check.sh", "runs/A_cand.lean"], capture_output=True, text=True)
    verdict = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
    print(f"{name:30s} -> {verdict}")
pathlib.Path("runs/A_cand.lean").unlink(missing_ok=True)
