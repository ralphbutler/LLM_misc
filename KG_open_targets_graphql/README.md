# Open Targets GraphQL Demo

This demo contains two parts which have been kept separate for now:

## 1. GraphQL Agent (`agent_graphql.py`)

This program reads `open_targets_schema.txt` and uses the schema to create valid GraphQL queries to the Open Targets Knowledge Graph. Features:

- Uses the new PydanticAI agent support to create the agent
- Schema can be downloaded from their [GraphQL Playground](https://api.platform.opentargets.org/api/v4/graphql/browser) (right-hand side)

## 2. Alzheimer's Analysis (`alz_both.py`) 

This program combines the functions of `alz1.py` and `alz2.py` to:

- Use GraphQL queries to obtain information from the Knowledge Graph
- Compute latent relationships
- Currently uses hard-coded queries rather than the agent above

## Background on Latent Relationships

The idea for exploring latent relationships came from Rick:

> It would be good to search this for latent relationships where there are
> connections that are transitive and not yet explicitly captured. A few years
> ago we did a version of this using MESH terms and prolog and could mine such
> relationships. This data might be better at doing that. The idea is to search
> for hypotheses for off target drugs that might be effective on some target.
