# Knowledge Graphs for RAG etc

Create a neo4j Knowledge Graph.
Demo creating it and using it with and without LangChain.

Such a KG could be used with an existing vector index, or the examples
show how you could also use the embedding support in neo4j as well.

- 00\_start\_neo4j\_docker.sh</br>
  start neo4j running in a docker container

- 01\_create\_kg.py</br>
    create a KG and add an index on embeddings and use the embeddings and simple queries via Cypher

- 02\_kg\_query\_langchain.py</br>
    sample query using LangChain facilities

- 03\_multihop\_qa.py</br>
    does a hop from one user to another to game nodes for second user