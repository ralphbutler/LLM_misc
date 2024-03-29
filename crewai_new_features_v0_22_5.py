
# use multiple agents to look for all 3 keys at one time
#   *** it is important to note that there are three copies of the dbsearch_agent and all
#   of them are executed in parallel (async) ***
# last agent uses pydantic to format results

## using anync may get an error message when trying to use a tool; turning off async
##    seems to fix it, so I assume crewai has a bug right now; may be some other issue
## using 3.5-turbo runs much faster and generally works fine

import sys, os, time
from textwrap import dedent

from langchain_openai import ChatOpenAI

from crewai import Agent, Task, Crew, Process
from crewai_tools import tool

from pydantic import BaseModel, Field
from typing import List

class KeyValuePair(BaseModel):
    key: str = Field(..., description="key into the database")
    val: int = Field(..., description="value for the key")  # note int

class NeedleInfo(BaseModel):
    kv_pair: List[KeyValuePair] = Field(..., description="list of keys and values")

## to use different GPT models  (gpt4 by default see print below)
model_name = "gpt-3.5-turbo"
# model_name = "gpt-4-turbo-preview"
# model_name = "gpt-4" # default model for agents inside crewai
llm = ChatOpenAI(model_name=model_name, temperature=0.0)

# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model='claude-3-opus-20240229')

fake_database = {
    "robot1": "this string contains the special value 111",
    "robot2": "this string contains the special value 222",
    "robot3": "this string contains the special value 333",
}

@tool("Database Search Tool")
def database_search_tool(key: str) -> str:
    """Search the database for the given key and retrieve the associated value."""
    print("\nDBGSRCH",key)
    import time, random
    time.sleep( 1.5 + random.random() )
    return fake_database[key]

def search_step_callback(arg):
    print("STEPCALLBACK") # ,arg)

def search_task_callback(arg):
    print("TASKCALLBACK") # ,arg)

robot1_agent = Agent(
    role="Database Searcher for Robot1",
    goal="Search database for key required by user",
    backstory=dedent("""\
        You are an expert at searching our database using a key to find an associated value."""
    ),
    verbose=True,
    allow_delegation=True,
    tools=[database_search_tool],
    llm=llm,
    step_callback=search_step_callback,
)

robot2_agent = Agent(
    role="Database Searcher for Robot2",
    goal="Search database for key required by user",
    backstory=dedent("""\
        You are an expert at searching our database using a key to find an associated value."""
    ),
    verbose=True,
    allow_delegation=True,
    tools=[database_search_tool],
    llm=llm,
    step_callback=search_step_callback,
)

robot3_agent = Agent(
    role="Database Searcher for Robot3",
    goal="Search database for key required by user",
    backstory=dedent("""\
        You are an expert at searching our database using a key to find an associated value."""
    ),
    verbose=True,
    allow_delegation=True,
    tools=[database_search_tool],
    llm=llm,
    step_callback=search_step_callback,
)

format_results_agent = Agent(
    role="Format Search Results For User",
    goal=dedent("""\
        Accept results from database searchers, and verify that the sought result was
        retrieved, and format the result as required by the user."""
    ),
    backstory=dedent("""\
        You were formerly a database searcher, but now have been promoted to format the
        results retrieved by new database searchers."""
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

robot1_task = Task(
    description="Search our database to retrieve the value for key: robot1.",
    expected_output="The specified key and value.",
    output_pydantic=NeedleInfo,
    agent=robot1_agent,
    async_execution=True,  ### <---- important ###
    callback=search_task_callback,
)

robot2_task = Task(
    description="Search our database to retrieve the value for key: robot2.",
    expected_output="The specified key and value.",
    output_pydantic=NeedleInfo,
    agent=robot2_agent,
    async_execution=True,  ### <---- important ###
    callback=search_task_callback,
)

robot3_task = Task(
    description="Search our database to retrieve the value for key: robot3.",
    expected_output="The specified key and value.",
    output_pydantic=NeedleInfo,
    agent=robot3_agent,
    async_execution=True,  ### <---- important ###
    callback=search_task_callback,
)

format_results_task = Task(
    description="Accept search results from all searchers, and format those results",
    expected_output=dedent("""\
        the keys and values for all three robots (robot1, robot2, robot3)"""
    ),
    # output_pydantic=NeedleInfo,
    output_json=NeedleInfo,
    agent=format_results_agent,
    context=[robot1_task,robot2_task,robot3_task],
)

crew = Crew(
    agents=[robot1_agent,robot2_agent,robot3_agent,format_results_agent],
    tasks=[robot1_task,robot2_task,robot3_task,format_results_task],
    verbose=2, # set it to 1 or 2 to different logging levels
    process=Process.hierarchical,
    manager_llm=llm,   # only used with hierarchical
)

result = crew.kickoff()
print(result)
