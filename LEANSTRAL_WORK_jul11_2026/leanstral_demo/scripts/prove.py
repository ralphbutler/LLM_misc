#!/usr/bin/env python3
"""
Drive Leanstral through the proving loop on a working file.
Usage: scripts/prove.py runs/B_<ts>.lean [max_rounds]

This model reasons verbosely and does not reliably emit one clean file. So we ask
it for ONLY the proof tactics (between sentinel markers) and splice them into the
pristine template ourselves — robust against any amount of rambling. Definitions
never change; only the proof is injected. Raw replies saved as runs/<stem>.rN.reply.txt.
"""
import sys, subprocess, json, urllib.request, urllib.error, re, pathlib, time, textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = "http://localhost:1234/v1/chat/completions"
MODEL = "leanstral-1.5-119b-a6b-mlx"
BEGIN, END = "###PROOF_BEGIN###", "###PROOF_END###"

SYSTEM = f"""You are Leanstral, an expert Lean 4 + Mathlib theorem prover.

Environment:
- Lean 4 (toolchain v4.20.0) with **Mathlib imported**. Use standard Lean 4 + Mathlib
  tactics and lemmas idiomatically (simp, simp_all, omega, decide, rcases, induction,
  Function.comp, the List/Function API, etc.). Prefer short, robust proofs.

You are given a Lean 4 file whose single theorem ends with `:= by` and a `sorry`.
Your job: produce the tactic proof that replaces the `sorry`.

OUTPUT CONTRACT (STRICT):
- Output ONLY the proof tactics — the lines that go after `:= by`.
- Do NOT restate the file, the imports, the definitions, the theorem signature, or `:= by`.
- Do NOT use code fences. Do NOT add prose.
- Wrap the tactics EXACTLY between a line containing {BEGIN} and a line containing {END}.
- Never use sorry, admit, axiom, or native_decide.

Example of the required format:
{BEGIN}
  induction s with
  | nil => rfl
  | cons b bs ih => simp [ih]
{END}
You may think first, but the FINAL thing you output must be one {BEGIN}/{END} block."""


def wait_until_ready(timeout=180):
    """After an MLX crash, LMStudio reloads the ~68GB model. Poll until it answers."""
    deadline = time.time() + timeout
    probe = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "ok"}],
                        "max_tokens": 1, "temperature": 0}).encode()
    while time.time() < deadline:
        time.sleep(8)
        try:
            req = urllib.request.Request(API, data=probe, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                json.load(r)
                print("  [model is back online]", flush=True)
                return True
        except Exception:
            continue
    return False


def call(messages, temperature=0.3, max_tokens=12000, retries=5):
    payload = {"model": MODEL, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, "stream": False}
    data = json.dumps(payload).encode()
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")[:400]
            print(f"  [HTTP {e.code}, attempt {attempt}/{retries}] {body}", flush=True)
            if attempt == retries:
                raise
            if "crash" in body.lower():
                print("  [model crashed; waiting for LMStudio to reload it...]", flush=True)
                wait_until_ready()
            else:
                time.sleep(5)
        except urllib.error.URLError as e:
            print(f"  [URLError, attempt {attempt}/{retries}] {e}", flush=True)
            if attempt == retries:
                raise
            time.sleep(5)


def extract_proof(text):
    """Take the tactics from the LAST sentinel block; fall back to the last
    fenced/plain block that is a proof fragment (no namespace/theorem).
    IMPORTANT: preserve the proof's relative indentation (do not lstrip the first
    line), then dedent uniformly so splice() can apply one clean base indent."""
    m = re.findall(re.escape(BEGIN) + r"(.*?)" + re.escape(END), text, re.S)
    if m:
        proof = m[-1]
    else:
        blocks = [b for b in re.findall(r"```(?:lean)?\n(.*?)```", text, re.S)
                  if b.strip() and "namespace" not in b and "theorem" not in b]
        if not blocks:
            return None
        proof = blocks[-1]
    proof = proof.strip("\n")                                   # drop surrounding blank lines only
    proof = re.sub(r"\A[ \t]*```(?:lean)?[ \t]*\n", "", proof)  # strip a wrapping fence if present
    proof = re.sub(r"\n[ \t]*```[ \t]*\Z", "", proof)
    proof = textwrap.dedent(proof).strip("\n")                  # normalize to column 0, keep structure
    if re.match(r"by(\s|$)", proof):                            # drop a leading 'by' if the model added it
        proof = re.sub(r"\Aby[ \t]*\n?", "", proof)
        proof = textwrap.dedent(proof).strip("\n")
    return proof or None


def splice(template, proof):
    """Inject the proof after `:= by`. Normalize indentation first (the model may
    already indent its tactics), then apply a single uniform 2-space indent."""
    m = re.search(r":=\s*by\s*\n\s*sorry", template)
    pre, post = template[:m.start()], template[m.end():]
    body = textwrap.dedent(proof.strip("\n"))
    indented = "\n".join(("  " + ln) if ln.strip() else ln for ln in body.splitlines())
    return pre + ":= by\n" + indented + "\n" + post


def check(f):
    p = subprocess.run([str(ROOT / "scripts" / "check.sh"), str(f)], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main():
    workfile = pathlib.Path(sys.argv[1])
    if not workfile.is_absolute():
        workfile = ROOT / workfile
    max_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    template = workfile.read_text()  # pristine copy; we always splice into this
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "Here is the Lean 4 file (Mathlib is imported). Prove the theorem: give ONLY the "
            f"tactics that replace `sorry`, wrapped between {BEGIN} and {END}.\n\n```lean\n"
            + template + "\n```"},
    ]
    for rnd in range(1, max_rounds + 1):
        print(f"\n===== ROUND {rnd}: calling Leanstral (temp={temperature}) =====", flush=True)
        reply = call(messages, temperature=temperature)
        (ROOT / "runs" / f"{workfile.stem}.r{rnd}.reply.txt").write_text(reply)
        proof = extract_proof(reply)
        if not proof:
            print("!! No proof block found in reply (raw saved). First 800 chars:\n")
            print(reply[:800]); return
        workfile.write_text(splice(template, proof))
        rc, out = check(workfile)
        print(out)
        if rc == 0:
            print(f"\n★ SOLVED in {rnd} round(s): {workfile}"); return
        messages.append({"role": "assistant", "content": f"{BEGIN}\n{proof}\n{END}"})
        out_t = out if len(out) <= 2600 else out[:500] + "\n...\n" + out[-2000:]
        messages.append({"role": "user", "content":
            "That proof did not compile. Here is the exact `lake env lean` output "
            "(line numbers are for the full spliced file):\n\n```\n" + out_t + "\n```\n\n"
            f"Return a corrected proof between {BEGIN} and {END} — tactics only, no file, no fences. "
            "Only Lean 4 + Mathlib; never use sorry/admit/native_decide."})
    print("\n✗ Not solved within the round budget.")


if __name__ == "__main__":
    main()
