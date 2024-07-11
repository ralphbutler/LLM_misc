
import sys, os, time
from routellm.controller import Controller

client = Controller(
  routers=["mf"],  # matrix factorization
  strong_model="claude-3-5-sonnet-20240620",
  weak_model="ollama/mistral",  # "ollama/gemma:latest",
)

# query = "What is the square root of 14 to 2 decimal places?"           # sonnet probably
query = "Which reflects light better, black paint or white paint?"   # ollama probably

# useful threshold values can be computed by running the calibrate_threshold program;
# routellm will compute a winrate for each prompt, and if that winrate exceeds the
#     user-specified threshold, then the strong model will be used; low threshold
#     values are easier to exceed, and thus favor using the strong model often
# threshold = 0.07075
# threshold = 0.1881
threshold = 0.11593

stime = time.time()
response = client.chat.completions.create(
    # model="router-mf-0.11593",
    model=f"router-mf-{threshold}",
    messages=[
        { "role": "user", "content":  query}
    ]
)
print(response.choices[0]["message"]["content"])
print("MODEL", response["model"])
print(f"time {time.time()-stime:0.2f} \n")
