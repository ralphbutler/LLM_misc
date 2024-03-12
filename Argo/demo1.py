
from ArgoLLM import ArgoLLM

llm = ArgoLLM(model_type='gpt35', temperature=0)

response = llm.invoke('Why is the sky blue?')
print(response)
print(type(response))
