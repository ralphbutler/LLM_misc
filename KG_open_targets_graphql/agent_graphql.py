import json, re, requests
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


with open("open_targets_schema.txt") as f:
    open_targets_schema = f.read()

SYSTEM_PROMPT=f"""
    You are a helpful agent that translates user's natural language queries into valid
    GraphQL queries to the Open Targets platform's Knowledge Graph.

    Follow the Open Targets schema here:
    <schema>
    {open_targets_schema}
    </schema>
"""

class GraphQlQuery(BaseModel):
    gql_query: str = Field(description="A valid GraphQL query to Open Targets Knowledge Graph")

graphql_agent = Agent(
    "openai:gpt-4o",
    deps_type=str,   # user query
    result_type=GraphQlQuery, # graphql query
    system_prompt=SYSTEM_PROMPT,
)

class GraphQlClient:

    def __init__(self):
        self.graphql_URL = "https://api.platform.opentargets.org/api/v4/graphql"
        
    def search_disease(self, disease_name):
        """Search for a disease and return its ID"""

        user_request = f"""
            Create a GraphQL query to retrieve the ID for this disease: {disease_name}.
        """
        response = graphql_agent.run_sync(user_request)
        print("QUERY TO SEND")
        print(response.data.gql_query)
        print("QUERY END")

        ## print("EXITING") ; exit(0)

        response = requests.post(self.graphql_URL, json={'query': response.data.gql_query})
        jdata = response.json()
        print("JSON RECVD")
        print(jdata)
        print("JSON RECVD END")

        hits = jdata["data"]["search"]["hits"]
        print("HITS")
        print(hits)
        print("HITS END")
        
        print("\nDisease search results:")
        for hit in jdata['data']['search']['hits']:
            print(f"ID: {hit['id']}, Name: {hit['name']}")
        
        return hits[0]['id']  # Return first disease ID

    def get_disease_targets(self, disease_id):
        """Get top targets associated with the disease"""

        user_request = f"""
            Create a GraphQL query to retrieve associated targets for disease with ID: {disease_id}.
        """
        response = graphql_agent.run_sync(user_request)
        print("QUERY TO SEND")
        print(response.data.gql_query)
        print("QUERY END")

        response = requests.post(self.graphql_URL, json={'query': response.data.gql_query})
        jdata = response.json()
        print("JSON RECVD")
        print(jdata)
        print("JSON RECVD END")

        print("\nTop disease-associated targets:")
        targets = []
        if 'data' in jdata and jdata['data']['disease']:
            for row in jdata['data']['disease']['associatedTargets']['rows']:
                target = row['target']
                score = row['score']
                print(f"Target: {target['approvedSymbol']}, Score: {score}")
                targets.append({
                    'id': target['id'],
                    'symbol': target['approvedSymbol'],
                    'score': score
                })

        return targets

    def get_drugs_for_target(self, target_id):
        """Get drugs that target a specific protein"""

        user_request = f"""
            Create a GraphQL query to retrieve drugs associated with target: {target_id}.
        """
        response = graphql_agent.run_sync(user_request)
        print("QUERY TO SEND")
        print(response.data.gql_query)
        print("QUERY END")

        response = requests.post(self.graphql_URL, json={'query': response.data.gql_query})
        jdata = response.json()
        print("JSON RECVD")
        print(jdata)
        print("JSON RECVD END")

        print("\nTop disease-associated targets:")
        drugs = []
        if 'data' in jdata and jdata['data']['target']['knownDrugs']:
            drugs = jdata['data']['target']['knownDrugs']['rows']

        return drugs

    def get_target_interactions(self, target_id):
        """Get protein-protein interactions for this target"""

        user_request = f"""
            Create a GraphQL query to retrieve interactions associated with target: {target_id}.
        """
        response = graphql_agent.run_sync(user_request)
        print("QUERY TO SEND")
        print(response.data.gql_query)
        print("QUERY END")

        response = requests.post(self.graphql_URL, json={'query': response.data.gql_query})
        jdata = response.json()
        print("JSON RECVD")
        print(jdata)
        print("JSON RECVD END")

        print("\nTop disease-associated targets:")
        interactions = []
        if 'data' in jdata and jdata['data']['target']['interactions']:
            interactions = jdata['data']['target']['interactions']['rows']

        return interactions


def main():
    gqclient = GraphQlClient()
    
    # Step 1: Find disease ID through search
    print("\nStep 1: Searching for disease...")
    disease_id = gqclient.search_disease("Alzheimer's disease")
    print(f"Using disease ID: {disease_id}")

    # Step 2: Get direct disease-associated targets
    print("\nStep 2: Getting primary disease-associated targets...")
    primary_targets = gqclient.get_disease_targets(disease_id)

    print("\nPrimary Disease-Associated Targets:")
    for target in primary_targets:
        print(f"- {target['symbol']} (ID: {target['id']})")

    # Step 3: For primary target find drugs and interactions

    target_id = primary_targets[0]

    print("\nStep 3drugs: Getting drugs for target",target_id)
    drugs = gqclient.get_drugs_for_target(target_id)
    for drug in drugs:
        print("DRUG",drug)
        break  ### just print first of perhaps many

    ## print("EXITING") ; exit(0)

    print("\nStep 3interactions: Getting interactions for target",target_id)
    interactions = gqclient.get_target_interactions(target_id)
    for interaction in interactions:
        print("INTERACTION",interaction)
        break  ### just print first of perhaps many


if __name__ == "__main__":
    main()
