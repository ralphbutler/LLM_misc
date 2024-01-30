
import sys, os, time

import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch

llm = dspy.OpenAI(model='gpt-3.5-turbo-1106', max_tokens=250)
dspy.settings.configure(lm=llm)

gms8k = GSM8K()
(trainset,devset) = (gms8k.train,gms8k.dev)
(trainset,devset) = (trainset[0:10],devset[0:20])
print(len(trainset), len(devset))

NUM_THREADS = 4
evaluate = Evaluate(
    devset=devset[:],
    metric=gsm8k_metric,
    num_threads=NUM_THREADS,
    display_progress=True,
    display_table=False,
)

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

class ReAct(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ReAct("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)


# these ought to be cmd-line args
DO_DOMPILE = True
LOAD_PREV_COMPILED = False
COT_OR_REACT = "COT"  # "REACT"

if DO_DOMPILE:
    # even smaller config values made it run so long that I killed it
    # orig config: 8, 8, 10
    config = dict(max_bootstrapped_demos=3, max_labeled_demos=3,
                  num_candidate_programs=3, num_threads=NUM_THREADS)
    # randomsearch increases the time even without config, but it seems tolerable
    # teleprompter = BootstrapFewShotWithRandomSearch(metric=gsm8k_metric, **config)
    # teleprompter = BootstrapFewShotWithRandomSearch(metric=gsm8k_metric)
    teleprompter = BootstrapFewShot(metric=gsm8k_metric)
    stime = time.time()
    print("COMPILING")
    if COT_OR_REACT == "COT":
        predictor = teleprompter.compile(CoT(), trainset=trainset, valset=devset)
    else:
        predictor = teleprompter.compile(ReAct(), trainset=trainset, valset=devset)
    print("COMPILE TIME FOR",COT_OR_REACT,time.time()-stime)
    predictor.save(f"{COT_OR_REACT}_compiled.json")
else:
    if COT_OR_REACT == "COT":
        predictor = CoT()
    else:
        predictor = ReAct()
    if LOAD_PREV_COMPILED:
        predictor.load(f"{COT_OR_REACT}_compiled.json")

print("DOING EVAL")
result = evaluate(predictor, devset=devset[:])  # may print quite a bit
print("EVAL RESULT", result)

result = llm.inspect_history(n=1)
print("HISTORY",result)
