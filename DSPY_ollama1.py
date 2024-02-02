import sys
import os

import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.hotpotqa import HotPotQA
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, BootstrapFinetune
import dsp.modules.ollama

mistral = dsp.modules.ollama.OllamaLocal(
    model="mistral",
)
mistral.base_url = "http://localhost:11434"

colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

# dspy.settings.configure(rm=colbertv2, lm=mistral)
dspy.settings.configure(lm=mistral)

predict = dspy.Predict('question -> answer')
try:   ### problem with ollama handling same query twice in a row ########
       # https://github.com/ollama/ollama/pull/2018
    x = predict(question="Who was the first president of the USA?")
except:
    x = predict(question="Who was the second president of the USA?")
print(x)

train = [('Who was the director of the 2009 movie featuring Peter Outerbridge as William Easton?', 'Kevin Greutert'),
         ('The heir to the Du Pont family fortune sponsored what wrestling team?', 'Foxcatcher'),
         ('In what year was the star of To Hell and Back born?', '1925'),
         ('Which award did the first book of Gary Zukav receive?', 'U.S. National Book Award'),
         ('What documentary about the Gilgo Beach Killer debuted on A&E?', 'The Killing Season'),
         ('Which author is English: John Braine or Studs Terkel?', 'John Braine'),
         ('Who produced the album that included a re-recording of "Lithium"?', 'Butch Vig')]

train = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in train]

class CoT(dspy.Module):  # let's define a new module
    def __init__(self):
        super().__init__()
        # here we declare the chain of thought sub-module, so we can later compile it (e.g., teach it a prompt)
        self.generate_answer = dspy.ChainOfThought('question -> answer')
    
    def forward(self, question):
        return self.generate_answer(question=question)  # here we use the module

metric_EM = dspy.evaluate.answer_exact_match

# more demos in this next line help
teleprompter = BootstrapFewShot(
    metric=metric_EM,
    # max_bootstrapped_demos=6,
    # max_rounds=2,  # fails with bug in code ? maybe just with open LLMs ?
)
cot_compiled = teleprompter.compile(CoT(), trainset=train)

x = cot_compiled("What is the capital of Germany?")
print(x)

