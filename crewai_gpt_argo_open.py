
import os
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

## to use different Ollama models
# model_name = "mistral"
# llm = Ollama(model=model_name)

## to use different GPT models  (gpt4 by default see print below)
# model_name = "gpt-3.5-turbo"
# model_name = "gpt-4-turbo-preview"
# model_name = "gpt-4" # default model for agents inside crewai
# llm = ChatOpenAI(model_name=model_name, temperature=0.0)

# ssh -D 32000 -CqN homes.cels.anl.gov    # <-- do this is separate window ###
from argo.ArgoLLM import ArgoLLM
os.environ["https_proxy"] = "socks5h://localhost:32000"
llm = ArgoLLM(model_type='gpt35', temperature=0)
response = llm.invoke("who was first USA president?")
print("DBGRESP",response)


chunk_of_text = """
This is the first sentence in paragraph 1.
This is the second sentence in paragraph 1.
This is the last sentence in paragraph 1.
This is the first sentence in paragraph 2.
This is the second sentence in paragraph 2.
This is the last sentence in paragraph 2.
This is the first sentence in paragraph 3.
This is the second sentence in paragraph 3.
This is the last sentence in paragraph 3.
"""
 
paragraph_parser = Agent(
    role="Senior Parser of Text",
    goal="Combine sentences from blocks of text into coherent paragraphs",
    backstory="""You are an expert at examining text to determine how to
    combine sentences from blocks of text into coherent paragraphs.""",
    verbose=False,   # chg for debugging
    allow_delegation=False,
    # tools=[search_tool],
    llm=llm,
)

task1 = Task(
    description="""Split up words and sentences sensibly to make coherent paragraphs that
    discuss a single idea or subject.  Ignore partial paragraphs at the start and end of 
    each chunk.  Here is the block of text to handle today: """ + chunk_of_text,
    expected_output="""A set of coherent paragraphs formed from sentences in the provided text.
    Your final answer MUST be ONLY a JSON array of paragraphs.  For example:
        [
            "Sentence 1 of paragraph 1. Sentence 2 of paragraph 1. Sentence 3 of paragraph 1.",
            "Sentence 1 of paragraph 2. Sentence 2 of paragraph 2. Sentence 3 of paragraph 2.",
            "Sentence 1 of paragraph 3. Sentence 2 of paragraph 3. Sentence 3 of paragraph 3.",
        ]
    """,
    agent=paragraph_parser,
)

crew = Crew(
  agents=[paragraph_parser],
  tasks=[task1],
  verbose=0, # set it to 1 or 2 to different logging levels
  process="sequential",  # default
  manager_llm=llm,
)

# print("DBGALLM",paragraph_parser.llm)
# print("DBGCLLM",crew.manager_llm)

result = crew.kickoff()

print("-"*50)
print(result)
