
from ArgoLLM import ArgoLLM

from langchain_core.prompts import ChatPromptTemplate

llm = ArgoLLM(model_type='gpt35', temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "Tell me a joke about {input}.")
    ]
)

chain = prompt | llm

response = chain.invoke( {"input": "dogs"} )
print(response)
print(type(response))
