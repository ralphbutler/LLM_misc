
# this program combines both alz1.y and alz2.py AND also puts all 
#   the functions into a class which can be used elsewhere

import json, re, requests
from collections import defaultdict

class GraphQlClient:

    def __init__(self):
        self.graphql_URL = "https://api.platform.opentargets.org/api/v4/graphql"
        
    def search_disease(self, disease_name):
        """Search for a disease and return its ID"""
        url = "https://api.platform.opentargets.org/api/v4/graphql"
        
        query = """
        {
          search(queryString: "Alzheimer's disease") {
            hits {
              id
              name
              entity
            }
          }
        }
        """
        
        response = requests.post(url, json={'query': query})
        data = response.json()
        
        print("\nDisease search results:")
        for hit in data['data']['search']['hits']:
            if hit['entity'] == 'disease':
                print(f"ID: {hit['id']}, Name: {hit['name']}")
        
        return data['data']['search']['hits'][0]['id']  # Return first disease ID

    def get_disease_targets(self, disease_id):
        """Get top targets associated with the disease"""

        query = """
        query($diseaseId: String!) {
          disease(efoId: $diseaseId) {
            id
            name
            associatedTargets(page: {index: 0, size: 10}) {
              rows {
                target {
                  id
                  approvedSymbol
                  approvedName
                }
                score
              }
            }
          }
        }
        """

        response = requests.post(self.graphql_URL,
                                 json={'query': query,
                                       'variables': {'diseaseId': disease_id}})
        data = response.json()

        print("\nTop disease-associated targets:")
        targets = []
        if 'data' in data and data['data']['disease']:
            for row in data['data']['disease']['associatedTargets']['rows']:
                target = row['target']
                score = row['score']
                print(f"Target: {target['approvedSymbol']}, Score: {score}")
                targets.append({
                    'id': target['id'],
                    'symbol': target['approvedSymbol'],
                    'score': score
                })

        return targets

    def get_target_interactions(self, target_id):
        """Get protein interactions for a target"""
        url = self.api_url
        
        # Modified query to ensure we're getting the right data
        query = """
        query($targetId: String!) {
          target(ensemblId: $targetId) {
            id
            approvedSymbol
            interactions {
              count
              rows {
                targetA {
                  id
                  approvedSymbol
                }
                targetB {
                  id
                  approvedSymbol
                }
                score
              }
            }
          }
        }
        """
        
        response = requests.post(url,
                               json={'query': query,
                                    'variables': {'targetId': target_id}})
        return response.json()

    def get_drugs_for_target(self, target_id):
        """Get drugs that target a specific protein"""
        url = self.api_url

        query = """
        query($targetId: String!) {
          target(ensemblId: $targetId) {
            id
            approvedSymbol
            knownDrugs(size: 10) {
              rows {
                drug {
                  id
                  name
                  maximumClinicalTrialPhase
                  mechanismsOfAction {
                    rows {
                      actionType
                    }
                  }
                }
                phase
                status
              }
            }
          }
        }
        """

        response = requests.post(url,
                                 json={'query': query,
                                       'variables': {'targetId': target_id}})
        return response.json()

    def get_extended_relationships(self, target_id):
        """Get extended relationships for a specific target"""
        url = "https://api.platform.opentargets.org/api/v4/graphql"
    
        print(f"  Fetching data from Open Targets API...")
        
        query = """
        query($targetId: String!) {
          target(ensemblId: $targetId) {
            id
            approvedSymbol
            knownDrugs {
              rows {
                drug {
                  id
                  name
                  maximumClinicalTrialPhase
                  mechanismsOfAction {
                    rows {
                      actionType
                    }
                  }
                }
              }
            }
            interactions {
              rows {
                targetB {
                  id
                  approvedSymbol
                  knownDrugs {
                    rows {
                      drug {
                        id
                        name
                        maximumClinicalTrialPhase
                      }
                    }
                  }
                }
                score
              }
            }
          }
        }
        """
        
        try:
            response = requests.post(url,
                                   json={'query': query,
                                        'variables': {'targetId': target_id}})
            data = response.json()
            
            # Debug print with more detail
            print(f"\nProcessing data for target {target_id}:")
            if 'data' in data and data['data']['target']:
                target_data = data['data']['target']
                print(f"Found target: {target_data['approvedSymbol']}")
                n_drugs = len(target_data.get('knownDrugs', {}).get('rows', []))
                n_interactions = len(target_data.get('interactions', {}).get('rows', []))
                print(f"Processing {n_drugs} direct drugs...")
                print(f"Processing {n_interactions} protein interactions...")
                print(f"Analyzing potential drug connections through interactions...")
            else:
                print("No target data found or error in response")
                print(f"Response: {data}")
                
            return data
            
        except Exception as e:
            print(f"Error getting extended relationships for {target_id}: {str(e)}")
            return None

    def analyze_extended_paths(self, data):
        findings = {
            'direct_drugs': set(),  # Using set for unique entries
            'indirect_drugs': set(),
            'potential_repurposing': []
        }

        if not data or 'data' not in data or not data['data']['target']:
            return findings

        target_data = data['data']['target']
        target_symbol = target_data['approvedSymbol']

        # Process direct drugs
        if 'knownDrugs' in target_data and target_data['knownDrugs']:
            for drug_row in target_data['knownDrugs']['rows']:
                if drug_row and 'drug' in drug_row:
                    findings['direct_drugs'].add((  # Tuple for set inclusion
                        target_symbol,
                        drug_row['drug']['name'],
                        drug_row['drug']['maximumClinicalTrialPhase']
                    ))

        # Track known relationships for later reference
        known_relationships = set()
        for drug_target in findings['direct_drugs']:
            known_relationships.add((drug_target[1], drug_target[0]))  # (drug, target) pairs

        # Process interactions (with additional error checking)
        if 'interactions' in target_data and target_data['interactions']:
            for interaction in target_data['interactions']['rows']:
                if not interaction:
                    continue

                secondary_target = interaction.get('targetB')
                if not secondary_target:
                    continue

                interaction_score = interaction.get('score', 0)

                if interaction_score > 0.5:
                    known_drugs = secondary_target.get('knownDrugs', {})
                    if not known_drugs:
                        continue

                    for drug_row in known_drugs.get('rows', []):
                        if not drug_row or 'drug' not in drug_row:
                            continue

                        drug = drug_row['drug']
                        if not drug or 'name' not in drug or 'maximumClinicalTrialPhase' not in drug:
                            continue

                        findings['indirect_drugs'].add((  # Tuple for uniqueness
                            drug['name'],
                            target_symbol,
                            secondary_target.get('approvedSymbol', 'Unknown'),
                            interaction_score,
                            drug['maximumClinicalTrialPhase']
                        ))

        # Identify repurposing opportunities (with known relationship check)
        seen_opportunities = set()
        for drug_info in findings['indirect_drugs']:
            drug_name, primary_target, secondary_target, score, phase = drug_info

            opportunity_key = (drug_name, primary_target)
            if (opportunity_key not in seen_opportunities and
                phase >= 2 and
                score >= 0.7):

                is_known = (drug_name, primary_target) in known_relationships

                findings['potential_repurposing'].append({
                    'drug': drug_name,
                    'current_target': secondary_target,
                    'proposed_target': primary_target,
                    'confidence_score': score * (phase / 4.0),
                    'is_known_relationship': is_known,
                    'rationale': f"{'Known ' if is_known else ''}High-phase drug ({phase}) with "
                               f"strong interaction score ({score:.2f}) through protein interaction network"
                })
                seen_opportunities.add(opportunity_key)

        return findings

    def identify_repurposing_opportunities(self, findings):
        """
        Identify potential drug repurposing opportunities based on network analysis
        """
        opportunities = []

        # Look for high-phase indirect drugs with strong interaction scores
        for drug_info in findings['indirect_drugs']:
            if (drug_info['phase'] >= 2 and
                drug_info['interaction_score'] >= 0.7 and
                drug_info not in [d['drug'] for d in findings['direct_drugs']]):

                opportunities.append({
                    'drug': drug_info['drug'],
                    'current_target': drug_info['secondary_target'],
                    'proposed_target': drug_info['primary_target'],
                    'confidence_score': drug_info['interaction_score'] *
                                      (drug_info['phase'] / 4.0),
                    'rationale': f"High-phase drug ({drug_info['phase']}) with strong "
                               f"interaction score ({drug_info['interaction_score']})"
                })

        return opportunities

    def get_extended_network(self, target_id):
        """Get extended network for a target including drugs and interactions"""
        url = "https://api.platform.opentargets.org/api/v4/graphql"
        
        query = """
        query($targetId: String!) {
          target(ensemblId: $targetId) {
            id
            approvedSymbol
            interactions {
              rows {
                targetB {
                  id
                  approvedSymbol
                  knownDrugs {
                    rows {
                      drug {
                        id
                        name
                        maximumClinicalTrialPhase
                        mechanismsOfAction {
                          rows {
                            actionType
                          }
                        }
                      }
                    }
                  }
                  interactions {
                    rows {
                      targetB {
                        id
                        approvedSymbol
                        knownDrugs {
                          rows {
                            drug {
                              id
                              name
                              maximumClinicalTrialPhase
                            }
                          }
                        }
                      }
                      score
                    }
                  }
                }
                score
              }
            }
          }
        }
        """
        
        response = requests.post(url, json={'query': query, 'variables': {'targetId': target_id}})
        return response.json()

    def find_latent_relationships(self):
        # Hard-coded values from our previous analysis
        psen1_id = "ENSG00000080815"  # PSEN1
        app_id = "ENSG00000142192"    # APP
        
        all_paths = defaultdict(list)
        seen_paths = set()
        
        for start_id, start_name in [(psen1_id, "PSEN1"), (app_id, "APP")]:
            print(f"\nAnalyzing network from {start_name}...")
            network = self.get_extended_network(start_id)
            
            if 'data' in network and network['data']['target']:
                target_data = network['data']['target']
                
                # First level interactions
                for interaction1 in target_data.get('interactions', {}).get('rows', []):
                    if not interaction1 or interaction1.get('score', 0) < 0.9:  # Very high confidence only
                        continue
                        
                    intermediate = interaction1.get('targetB')
                    if not intermediate:
                        continue
                    
                    # Second level interactions
                    for interaction2 in intermediate.get('interactions', {}).get('rows', []):
                        if not interaction2 or interaction2.get('score', 0) < 0.9:
                            continue
                            
                        final_target = interaction2.get('targetB')
                        if not final_target:
                            continue
                        
                        # Look for high-phase drugs
                        for drug_row in final_target.get('knownDrugs', {}).get('rows', []):
                            if not drug_row or 'drug' not in drug_row:
                                continue
                                
                            drug = drug_row['drug']
                            phase = drug.get('maximumClinicalTrialPhase', 0)
                            
                            if phase >= 3:  # Only Phase 3 and 4
                                path_key = f"{start_name}-{intermediate['approvedSymbol']}-{final_target['approvedSymbol']}-{drug['name']}"
                                
                                if path_key not in seen_paths:
                                    path_info = {
                                        'start': start_name,
                                        'intermediate': intermediate['approvedSymbol'],
                                        'end_target': final_target['approvedSymbol'],
                                        'drug': drug['name'],
                                        'phase': phase,
                                        'confidence': min(interaction1['score'], interaction2['score'])
                                    }
                                    all_paths[phase].append(path_info)
                                    seen_paths.add(path_key)
        
        # Report findings
        print("\nMost Promising Latent Relationships:")
        print("==================================")
        
        total_paths = sum(len(paths) for paths in all_paths.values())
        print(f"\nFound {total_paths} unique high-confidence paths")
        
        for phase in sorted(all_paths.keys(), reverse=True):
            paths = all_paths[phase]
            # Sort by confidence and take top 3
            top_paths = sorted(paths, key=lambda x: x['confidence'], reverse=True)[:3]
            
            print(f"\nTop Phase {phase} Paths:")
            print("-" * 30)
            
            for path in top_paths:
                print(f"\n{path['start']} -> {path['intermediate']} -> {path['end_target']}")
                print(f"Drug: {path['drug']}")
                print(f"Confidence: {path['confidence']:.2f}")


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

    # Step 3: For each primary target, get extended relationships
    print("\nStep 3: Analyzing extended relationships for each target...")
    all_findings = {
        'direct_drugs': set(),
        'indirect_drugs': set(),
        'potential_repurposing': []
    }

    for target in primary_targets:
        print(f"\nAnalyzing extended relationships for {target['symbol']}...")
        relationship_data = gqclient.get_extended_relationships(target['id'])
        target_findings = gqclient.analyze_extended_paths(relationship_data)

        # Merge findings
        all_findings['direct_drugs'].update(target_findings['direct_drugs'])
        all_findings['indirect_drugs'].update(target_findings['indirect_drugs'])
        all_findings['potential_repurposing'].extend(target_findings['potential_repurposing'])

    # Step 4: Report comprehensive findings
    print("\nStep 4: Final Analysis Results")

    print("\n=== DIRECT RELATIONSHIPS ===")
    print("Primary Targets and Their Direct Drugs:")
    for target, drug, phase in sorted(all_findings['direct_drugs']):
        print(f"- Target: {target}")
        print(f"  Drug: {drug} (Phase: {phase})")

    print("\n=== EXTENDED RELATIONSHIPS ===")
    print("Promising Indirect Drugs:")
    seen_paths = set()
    for drug, primary, secondary, score, phase in sorted(all_findings['indirect_drugs']):
        path_key = (drug, primary, secondary)
        if score >= 0.7 and phase >= 2 and path_key not in seen_paths:
            print(f"- {drug}")
            print(f"  Path: {primary} -> {secondary}")
            print(f"  Interaction Score: {score:.2f}")
            print(f"  Clinical Phase: {phase}")
            seen_paths.add(path_key)

    print("\n=== REPURPOSING OPPORTUNITIES ===")
    # Sort by confidence score in descending order
    sorted_opportunities = sorted(
        all_findings['potential_repurposing'],
        key=lambda x: x['confidence_score'],
        reverse=True
    )

    seen_drugs = set()
    for opportunity in sorted_opportunities:
        drug_key = (opportunity['drug'], opportunity['proposed_target'])
        if drug_key not in seen_drugs:
            print(f"\nDrug: {opportunity['drug']}")
            print(f"Current Target: {opportunity['current_target']}")
            print(f"Proposed Target: {opportunity['proposed_target']}")
            print(f"Confidence Score: {opportunity['confidence_score']:.2f}")
            if opportunity.get('is_known_relationship'):
                print("Status: Known relationship")
            print(f"Rationale: {opportunity['rationale']}")
            seen_drugs.add(drug_key)

    # Step 5: Finding latent relationships
    print("\nStep 5: Finding latent relationships...")
    print("\n=== FINDING LATENT RELATIONSHIPS ===")
    gqclient.find_latent_relationships()   # also calls get_extended_network()

if __name__ == "__main__":
    main()
