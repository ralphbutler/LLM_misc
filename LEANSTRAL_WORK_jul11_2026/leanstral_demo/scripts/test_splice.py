import importlib.util, pathlib, subprocess
spec = importlib.util.spec_from_file_location("p", "scripts/prove.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)

template = pathlib.Path("problems/B_RevComp.lean").read_text()

# Simulate a model reply: verbose rambling + the real answer in a sentinel block.
good_proof = """have hc : ∀ b : Base, comp (comp b) = b := by
  intro b; cases b with | mk sym masked => cases sym <;> rfl
unfold revcomp
simp only [List.map_reverse, List.reverse_reverse, List.map_map]
rw [show comp ∘ comp = id from funext hc, List.map_id]"""
fake_reply = f"Let me think... blah blah\n```lean\nsome junk fragment\n```\nMore reasoning.\n{p.BEGIN}\n{good_proof}\n{p.END}\ntrailing prose."

proof = p.extract_proof(fake_reply)
print("extracted proof matches:", proof.strip() == good_proof.strip())
out = pathlib.Path("runs/B_splicetest.lean")
out.write_text(p.splice(template, proof))
r = subprocess.run(["scripts/check.sh", "runs/B_splicetest.lean"], capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1])
