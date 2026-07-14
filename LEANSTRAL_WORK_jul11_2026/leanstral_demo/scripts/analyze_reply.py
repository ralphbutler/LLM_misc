import re, sys, glob
r = sorted(glob.glob("runs/B_*.r1.reply.txt"))[-1]
t = open(r).read()
print("reply:", r, "| chars:", len(t), "| lines:", t.count("\n"))
fence = "`" * 3
blocks = re.findall(fence + r"(?:lean)?\s*(.*?)" + fence, t, re.S)
print("total code blocks:", len(blocks))
for i, b in enumerate(blocks):
    print(f"  block {i}: {len(b):6d} chars  namespace={'namespace' in b}  "
          f"thm={b.count('theorem revcomp')}  sorry={'sorry' in b}  import={'import Mathlib' in b}")
# show head of the model's prose before first fence
head = t.split(fence)[0]
print("=== text before first code fence (first 400 chars) ===")
print(head[:400])
