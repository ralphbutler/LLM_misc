
# from the notebook page
#    What DSPy Modules are currently built-in?
#        dspy.Predict:
#        dspy.ChainOfThought:
#        dspy.ProgramOfThought:
#        dspy.ReAct:
#        dspy.MultiChainComparison:
#      We also have some function-style modules:
#        dspy.majority:

### all modules use Predict internally <-- ********

import dspy

lm = dspy.OpenAI(
    model='gpt-3.5-turbo-1106',
    max_tokens=300,
)
dspy.configure(lm=lm)

sentence = "it's a charming and often affecting journey."
# RMB: chgd sentence to comment  and  sentiment to feelings for demo
classify = dspy.Predict('comment -> feelings')
response = classify(sentence=sentence)
print("PREDICT")
print(response.feelings)   # and so have to use feelings here also
print(f"\n{'-'*50}\n")


question = "What's something great about the ColBERT retrieval model?"
classify = dspy.ChainOfThought('question -> answer', n=5)
response = classify(question=question)
print("CoT")
print(response.answer)
print("\nRATIONALE for CoT answer")
print(response.rationale)
print(f"\n{'-'*50}\n")


question = "What's something great about the ColBERT retrieval model?"
classify = dspy.ReAct('question -> answer')
response = classify(question=question)
print("ReAct")
print(response.answer)
print(f"\n{'-'*50}\n")
