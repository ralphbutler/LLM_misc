
# %%
import sys, os, time, csv

from langchain_community.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

# %%
url = "bolt://localhost:7687"
username ="neo4j"
password = "foobar_neo4j"

# %%
graph = Neo4jGraph(
    url=url, 
    username=username, 
    password=password
)

# using GraphCypherQAChain we can query using natural language
# %%

llm_query = ChatOpenAI(temperature=0)
chain = GraphCypherQAChain.from_llm( llm_query, graph=graph, verbose=False,) # True -> lots of output

# %%
# query = "Who are the friends for User with userid user_0001?"  # Friend_17
query = "What are all the games of the friend of the User with userid user_0001?"
response = chain.invoke(query)
print(response)
print("-"*50)

print("*** EXITING") ; exit(0)
