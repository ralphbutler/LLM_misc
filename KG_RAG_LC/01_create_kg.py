
# %%
import sys, os, time, csv

from langchain_community.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
# from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chains import GraphCypherQAChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI

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

social_group = {}
with open('social_data.tsv', 'r') as file:
    csv_reader = csv.reader(file, delimiter='\t')
    header = next(csv_reader)
    for row in csv_reader:
        userid = row[0]
        name = row[1]
        games = row[2].split(', ')
        friends = row[3].split(', ')
        social_group[userid] = {
            "name": name,
            "games": games,
            "friends": friends
        }
# print(social_group)

# %%
embeddings = OpenAIEmbeddings()
for (idx,userid) in enumerate(social_group):
    userinfo = social_group[userid]
    name = userinfo["name"]
    games = userinfo["games"]
    str_games = " ".join(games)
    embed_games = embeddings.embed_query(str_games)
    user_isrt_cmd = f"""
        MERGE (user:User {{userid: "{userid}", name: "{name}",
                           embed_games: {embed_games}}})
        """
    graph.query(user_isrt_cmd)
    for game in games:
        game_isrt_cmd = f"""
            MERGE (game:Game {{name: "{game}"}})
            """
        graph.query(game_isrt_cmd)
        relationship_isrt_cmd = f"""
            MATCH (user:User {{name: "{name}"}})
            MATCH (game:Game {{name: "{game}"}})
            MERGE (user)-[:ENJOYS]->(game)
            """
        graph.query(relationship_isrt_cmd)
    print(f"{idx}")

# RMB: note backticks in next cmd
index_create_cmd = """
    CREATE VECTOR INDEX index_user_embeddings  IF NOT EXISTS
        FOR (user:User)
        ON user.embed_games
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: "cosine"
    }}
    """
graph.query(index_create_cmd)

result = graph.query(
    """SHOW INDEXES
       YIELD name, type, labelsOrTypes, properties, options
       WHERE type = 'VECTOR'
    """
)
print("NUMIDXS",len(result))
print("all indexes:")
for x in result:
    print(x)

# search for user with similar games (see example* file)
find_a_friend_query = """
    WITH "user_0001" AS tUserid
    MATCH (tUser:User {userid: tUserid}), (cUser:User)
    WHERE NOT (tUser)-[:FRIEND_OF]-(cUser)  AND
          tUser.userid <> cUser.userid      
    WITH  tUser, cUser,
          gds.similarity.cosine(tUser.embed_games, cUser.embed_games) AS simval
    ORDER BY simval DESC
    RETURN tUser.name, cUser.name, simval
    LIMIT 2
"""
result = graph.query(find_a_friend_query)
result = result[0]
print(result)

## RMB TO HERE
# put the best friend candidate into the graph ###########################
tname = result["tUser.name"]
cname = result["cUser.name"]
relationship_isrt_cmd = f"""
    WITH "{tname}" as tname, "{cname}" as cname
    MATCH (tuser:User {{name: "{tname}"}})
    MATCH (cuser:User {{name: "{cname}"}})
    MERGE (tuser)-[:FRIEND]-(cuser)
    """
graph.query(relationship_isrt_cmd)
