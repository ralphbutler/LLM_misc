#!/usr/bin/env python3
"""
3-SAT Foundation: Complete implementation with MCMC + Neural Network
Phase 1: SAT instance representation and evaluation
Phase 2: Bit-flip neighborhood system
Phase 3: MCMC Sampling Layer (Algorithm 1 from paper)
Phase 4: Neural Network Integration for End-to-End Learning
"""

import random
import math
from typing import List, Tuple
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class SATInstance:
    """Represents a 3-SAT problem instance"""
    
    def __init__(self, num_vars: int, clauses: List[Tuple[int, int, int]]):
        """
        Initialize SAT instance
        Args:
            num_vars: Number of boolean variables (1 to num_vars)
            clauses: List of 3-tuples, each containing 3 literals
                    Positive literal i means variable i is true
                    Negative literal -i means variable i is false
                    Variables are 1-indexed (1, 2, ..., num_vars)
        """
        self.num_vars = num_vars
        self.clauses = clauses
        
        print(f"Created SAT instance: {num_vars} variables, {len(clauses)} clauses")
        print(f"First few clauses: {clauses[:3]}")
        
    def is_clause_satisfied(self, clause: Tuple[int, int, int], assignment: List[bool]) -> bool:
        """
        Check if a single clause is satisfied by the assignment
        Args:
            clause: 3-tuple of literals (can be positive or negative)
            assignment: List of boolean values (0-indexed: assignment[0] = value of variable 1)
        Returns:
            True if clause is satisfied, False otherwise
        """
        lit1, lit2, lit3 = clause
        
        # Convert 1-indexed literals to 0-indexed variable checks
        def literal_satisfied(literal):
            if literal > 0:
                # Positive literal: variable must be True
                return assignment[literal - 1] == True
            else:
                # Negative literal: variable must be False
                return assignment[abs(literal) - 1] == False
        
        # Clause is satisfied if ANY literal is satisfied (OR operation)
        return literal_satisfied(lit1) or literal_satisfied(lit2) or literal_satisfied(lit3)
    
    def evaluate(self, assignment: List[bool]) -> int:
        """
        Evaluate how many clauses are satisfied by the assignment
        Args:
            assignment: List of boolean values for variables
        Returns:
            Number of satisfied clauses (0 to len(clauses))
        """
        if len(assignment) != self.num_vars:
            raise ValueError(f"Assignment length {len(assignment)} != num_vars {self.num_vars}")
            
        satisfied_count = 0
        for i, clause in enumerate(self.clauses):
            if self.is_clause_satisfied(clause, assignment):
                satisfied_count += 1
                
        return satisfied_count
    
    def is_satisfiable(self, assignment: List[bool]) -> bool:
        """Check if assignment satisfies ALL clauses"""
        return self.evaluate(assignment) == len(self.clauses)
    
    def print_summary(self):
        """Print a summary of the SAT instance"""
        print(f"\n=== SAT Instance Summary ===")
        print(f"Variables: {self.num_vars}")
        print(f"Clauses: {len(self.clauses)}")
        print(f"Clause/Variable ratio: {len(self.clauses) / self.num_vars:.2f}")
        
        # Show variable frequency analysis
        var_pos_count = [0] * self.num_vars
        var_neg_count = [0] * self.num_vars
        
        for clause in self.clauses:
            for literal in clause:
                if literal > 0:
                    var_pos_count[literal - 1] += 1
                else:
                    var_neg_count[abs(literal) - 1] += 1
        
        print(f"Variable frequency ranges:")
        print(f"  Positive literals: {min(var_pos_count)} to {max(var_pos_count)}")
        print(f"  Negative literals: {min(var_neg_count)} to {max(var_neg_count)}")


class BitFlipNeighborhood:
    """Implements bit-flip neighborhood systems for boolean assignments"""
    
    def __init__(self, hamming_distance: int = 1):
        """
        Initialize neighborhood system
        Args:
            hamming_distance: How many bits to flip (1 = flip 1 bit, 2 = flip 2 bits, etc.)
        """
        self.distance = hamming_distance
        print(f"Created BitFlipNeighborhood with Hamming distance = {hamming_distance}")
        
    def get_neighbors(self, assignment: List[bool]) -> List[List[bool]]:
        """
        Get all neighbors at the specified Hamming distance
        Args:
            assignment: Current boolean assignment
        Returns:
            List of all neighboring assignments
        """
        neighbors = []
        n = len(assignment)
        
        if self.distance == 1:
            # Flip each bit individually
            for i in range(n):
                neighbor = assignment.copy()
                neighbor[i] = not neighbor[i]  # Flip bit i
                neighbors.append(neighbor)
                
        elif self.distance == 2:
            # Flip each pair of bits
            for i in range(n):
                for j in range(i + 1, n):
                    neighbor = assignment.copy()
                    neighbor[i] = not neighbor[i]  # Flip bit i
                    neighbor[j] = not neighbor[j]  # Flip bit j
                    neighbors.append(neighbor)
        else:
            print(f"Warning: Hamming distance {self.distance} not implemented, returning empty list")
            return []
        
        print(f"Generated {len(neighbors)} neighbors for assignment {assignment}")
        return neighbors
    
    def sample_random_neighbor(self, assignment: List[bool]) -> List[bool]:
        """
        Sample one neighbor uniformly at random
        Args:
            assignment: Current boolean assignment
        Returns:
            One randomly selected neighboring assignment
        """
        n = len(assignment)
        
        if self.distance == 1:
            # Efficiently sample one random 1-bit flip without generating all neighbors
            flip_idx = random.randint(0, n - 1)
            neighbor = assignment.copy()
            neighbor[flip_idx] = not neighbor[flip_idx]
            return neighbor
            
        elif self.distance == 2:
            # Efficiently sample one random 2-bit flip
            flip_indices = random.sample(range(n), 2)
            neighbor = assignment.copy()
            neighbor[flip_indices[0]] = not neighbor[flip_indices[0]]
            neighbor[flip_indices[1]] = not neighbor[flip_indices[1]]
            return neighbor
            
        else:
            print(f"Warning: Hamming distance {self.distance} not implemented, returning copy")
            return assignment.copy()
    
    def get_neighbor_count(self, num_vars: int) -> int:
        """
        Calculate how many neighbors an assignment has
        Args:
            num_vars: Number of variables
        Returns:
            Number of neighbors for any assignment
        """
        if self.distance == 1:
            return num_vars
        elif self.distance == 2:
            return num_vars * (num_vars - 1) // 2
        else:
            raise NotImplementedError(f"Hamming distance {self.distance} not implemented")


class MCMCSATLayer:
    """
    MCMC Sampling Layer implementing Algorithm 1 from the paper
    Uses Metropolis-Hastings to sample from Gibbs distribution over SAT assignments
    """
    
    def __init__(self, sat_instance: SATInstance, neighborhood: BitFlipNeighborhood, 
                 temperature: float = 1.0, constraint_penalty: float = 10.0):
        """
        Initialize MCMC layer
        Args:
            sat_instance: The SAT problem to solve
            neighborhood: Bit-flip neighborhood system
            temperature: Temperature parameter t (higher = more exploration)
            constraint_penalty: Penalty weight for unsatisfied clauses
        """
        self.sat_instance = sat_instance
        self.neighborhood = neighborhood
        self.temperature = temperature
        self.constraint_penalty = constraint_penalty
        
        print(f"Created MCMC SAT Layer:")
        print(f"  Temperature: {temperature}")
        print(f"  Constraint penalty: {constraint_penalty}")
        print(f"  Neighborhood: {neighborhood.distance}-bit flips")
    
    def objective_function(self, assignment: List[bool], logits: List[float]) -> float:
        """
        Compute ⟨ω,y⟩ + ψ(y) from the paper
        Args:
            assignment: Boolean assignment y
            logits: Learned logits ω from neural network
        Returns:
            Objective value (higher = better)
        """
        # ⟨ω,y⟩ term: learned preference for assignment
        learned_score = sum(logits[i] * (1.0 if assignment[i] else 0.0) 
                           for i in range(len(assignment)))
        
        # ψ(y) term: constraint satisfaction bonus
        satisfied_clauses = self.sat_instance.evaluate(assignment)
        constraint_score = self.constraint_penalty * satisfied_clauses
        
        total_score = learned_score + constraint_score
        return total_score
    
    def sample_mcmc(self, logits: List[float], num_iterations: int = 100, 
                   initial_assignment: List[bool] = None, verbose: bool = False) -> List[List[bool]]:
        """
        Run MCMC sampling (Algorithm 1 from paper)
        Args:
            logits: Learned logits ω ∈ ℝ^n from neural network
            num_iterations: Number of MCMC steps K
            initial_assignment: Starting assignment y^(0)
            verbose: Print detailed sampling steps
        Returns:
            List of samples [y^(1), y^(2), ..., y^(K)]
        """
        if len(logits) != self.sat_instance.num_vars:
            raise ValueError(f"Logits length {len(logits)} != num_vars {self.sat_instance.num_vars}")
        
        # Initialize chain
        if initial_assignment is None:
            current = [random.choice([True, False]) for _ in range(self.sat_instance.num_vars)]
        else:
            current = initial_assignment.copy()
        
        current_score = self.objective_function(current, logits)
        samples = []
        
        if verbose:
            print(f"\nMCMC Sampling: {num_iterations} iterations")
            print(f"Initial assignment: {current}")
            print(f"Initial score: {current_score:.2f}")
            print(f"Initial satisfaction: {self.sat_instance.evaluate(current)}/{len(self.sat_instance.clauses)}")
        
        # MCMC iterations
        for k in range(num_iterations):
            # Sample a neighbor (proposal step) - directly sample without generating all neighbors
            if self.neighborhood.distance == 1:
                # Efficiently sample one random 1-bit flip
                flip_idx = random.randint(0, len(current) - 1)
                proposal = current.copy()
                proposal[flip_idx] = not proposal[flip_idx]
            elif self.neighborhood.distance == 2:
                # Efficiently sample one random 2-bit flip
                flip_indices = random.sample(range(len(current)), 2)
                proposal = current.copy()
                proposal[flip_indices[0]] = not proposal[flip_indices[0]]
                proposal[flip_indices[1]] = not proposal[flip_indices[1]]
            else:
                # Fallback to the neighborhood method
                proposal = self.neighborhood.sample_random_neighbor(current)
            
            proposal_score = self.objective_function(proposal, logits)
            
            # Compute acceptance probability (Metropolis-Hastings)
            score_diff = proposal_score - current_score
            if score_diff >= 0:
                # Always accept improvements
                accept_prob = 1.0
            else:
                # Accept worse moves with probability exp(Δ/t)
                accept_prob = math.exp(score_diff / self.temperature)
            
            # Accept or reject
            if random.random() < accept_prob:
                current = proposal
                current_score = proposal_score
                accepted = True
            else:
                accepted = False
            
            samples.append(current.copy())
            
            if verbose and (k < 10 or k % 20 == 19):
                satisfaction = self.sat_instance.evaluate(current)
                print(f"  Step {k+1}: score={current_score:.2f}, sat={satisfaction}/{len(self.sat_instance.clauses)}, " +
                      f"accept_prob={accept_prob:.3f}, {'ACCEPT' if accepted else 'REJECT'}")
        
        return samples
    
    def forward(self, logits: List[float], num_iterations: int = 100, 
               initial_assignment: List[bool] = None, burn_in: int = 0) -> List[float]:
        """
        Compute expectation E[Y] under Gibbs distribution (the main layer output)
        Args:
            logits: Learned logits ω from neural network
            num_iterations: Number of MCMC samples
            initial_assignment: Starting assignment
            burn_in: Number of initial samples to discard
        Returns:
            Expected assignment [E[Y_1], E[Y_2], ..., E[Y_n]] ∈ [0,1]^n
        """
        # Run MCMC sampling
        samples = self.sample_mcmc(logits, num_iterations + burn_in, initial_assignment)
        
        # Discard burn-in samples
        if burn_in > 0:
            samples = samples[burn_in:]
        
        # Compute expectation
        num_vars = self.sat_instance.num_vars
        expectations = []
        
        for var_idx in range(num_vars):
            # E[Y_i] = average value of variable i across samples
            avg_value = sum(1.0 if sample[var_idx] else 0.0 for sample in samples) / len(samples)
            expectations.append(avg_value)
        
        return expectations
    
    def get_best_assignment(self, logits: List[float], num_iterations: int = 100) -> Tuple[List[bool], float, int]:
        """
        Find the best assignment encountered during MCMC sampling
        Args:
            logits: Learned logits ω
            num_iterations: Number of MCMC samples
        Returns:
            (best_assignment, best_score, satisfied_clauses)
        """
        samples = self.sample_mcmc(logits, num_iterations)
        
        best_assignment = None
        best_score = float('-inf')
        best_satisfaction = 0
        
        for sample in samples:
            score = self.objective_function(sample, logits)
            satisfaction = self.sat_instance.evaluate(sample)
            
            if score > best_score:
                best_assignment = sample.copy()
                best_score = score
                best_satisfaction = satisfaction
        
        return best_assignment, best_score, best_satisfaction


def load_dimacs_file(filename: str) -> SATInstance:
    """
    Load a SAT instance from DIMACS CNF format file (original function)
    Args:
        filename: Path to the DIMACS file
    Returns:
        SATInstance object
    """
    print(f"Loading DIMACS file: {filename}")
    
    num_vars = 0
    num_clauses = 0
    clauses = []
    
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip comment lines (start with 'c')
                if line.startswith('c'):
                    print(f"  Comment: {line}")
                    continue
                
                # Parse problem line (starts with 'p')
                if line.startswith('p'):
                    parts = line.split()
                    if len(parts) >= 4 and parts[1] == 'cnf':
                        num_vars = int(parts[2])
                        num_clauses = int(parts[3])
                        print(f"  Problem: {num_vars} variables, {num_clauses} clauses")
                    continue
                
                # Parse clause line (ends with 0)
                if line.endswith(' 0'):
                    # Remove the trailing 0 and split into integers
                    literals = [int(x) for x in line.split()[:-1]]
                    
                    if len(literals) == 3:
                        clauses.append(tuple(literals))
                    else:
                        print(f"  Warning: Non-3-SAT clause on line {line_num}: {literals}")
                        # Still add it, but note it's not standard 3-SAT
                        clauses.append(tuple(literals))
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        sys.exit(1)
    
    print(f"Successfully loaded: {len(clauses)} clauses")
    
    # Verify we got the expected number of clauses
    if len(clauses) != num_clauses:
        print(f"  Warning: Expected {num_clauses} clauses, got {len(clauses)}")
    
    return SATInstance(num_vars, clauses)


def load_dimacs_with_solution(filename: str) -> Tuple[SATInstance, List[bool]]:
    """
    Load a SAT instance from DIMACS CNF format file with solution from comment
    Args:
        filename: Path to the DIMACS file
    Returns:
        (SATInstance object, satisfying_assignment)
    """
    print(f"Loading DIMACS file with solution: {filename}")
    
    num_vars = 0
    num_clauses = 0
    clauses = []
    solution_literals = None
    
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Parse solution from comment line (starts with 'c SAT')
                if line.startswith('c SAT '):
                    # Extract solution: "c SAT -1 2 3 4 -5 ..." -> [-1, 2, 3, 4, -5, ...]
                    parts = line.split()[2:]  # Skip 'c' and 'SAT'
                    if parts and parts[-1] == '0':
                        parts = parts[:-1]  # Remove trailing 0
                    
                    solution_literals = [int(x) for x in parts]
                    print(f"  Found solution literals: {solution_literals[:5]}... (total: {len(solution_literals)})")
                    continue
                
                # Skip other comment lines
                if line.startswith('c'):
                    continue
                
                # Parse problem line (starts with 'p')
                if line.startswith('p'):
                    parts = line.split()
                    if len(parts) >= 4 and parts[1] == 'cnf':
                        num_vars = int(parts[2])
                        num_clauses = int(parts[3])
                        print(f"  Problem: {num_vars} variables, {num_clauses} clauses")
                    continue
                
                # Parse clause line (ends with 0)
                if line.endswith(' 0'):
                    literals = [int(x) for x in line.split()[:-1]]
                    if len(literals) >= 1:  # Allow non-3-SAT clauses
                        clauses.append(tuple(literals))
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None, None
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        return None, None
    
    # Convert solution literals to boolean array
    if solution_literals is None:
        print(f"  Warning: No solution found in comments")
        return None, None
    
    if num_vars == 0:
        print(f"  Warning: Problem size not found")
        return None, None
    
    # Initialize solution array - all False by default
    solution = [False] * num_vars
    
    # Set variables based on literals in solution
    for literal in solution_literals:
        if literal > 0:
            # Positive literal: set variable to True
            if literal <= num_vars:
                solution[literal - 1] = True
            else:
                print(f"  Warning: Literal {literal} exceeds num_vars {num_vars}")
        elif literal < 0:
            # Negative literal: set variable to False (already False by default)
            var_index = abs(literal) - 1
            if var_index < num_vars:
                solution[var_index] = False
            else:
                print(f"  Warning: Literal {literal} exceeds num_vars {num_vars}")
    
    print(f"Successfully loaded: {len(clauses)} clauses, solution: {solution[:5]}...")
    
    sat_instance = SATInstance(num_vars, clauses)
    
    # Verify solution is correct
    satisfaction = sat_instance.evaluate(solution)
    if satisfaction == len(clauses):
        print(f"  ✅ Solution verified: {satisfaction}/{len(clauses)} clauses satisfied")
    else:
        print(f"  ❌ Solution verification failed: {satisfaction}/{len(clauses)} clauses satisfied")
        return None, None
    
    return sat_instance, solution


def load_training_data(data_dir: str, max_files: int = None) -> List[Tuple[SATInstance, List[bool]]]:
    """
    Load training data from directory of DIMACS files with solutions
    Args:
        data_dir: Directory containing DIMACS files
        max_files: Maximum number of files to load (None for all)
    Returns:
        List of (sat_instance, satisfying_assignment) pairs
    """
    print(f"Loading training data from {data_dir}/")
    
    training_data = []
    
    try:
        # Get all files in directory first
        all_files = os.listdir(data_dir)
        print(f"All files in directory: {all_files[:10]}...")  # Show first 10 for debugging
        
        # More permissive file filtering - accept most files except obvious non-data files
        files = [f for f in all_files if not f.startswith('.') and not f.endswith('.py')]
        print(f"After filtering: found {len(files)} potential data files")
        
        if max_files:
            files = files[:max_files]
        
        print(f"Will process {len(files)} files")
        
        for i, filename in enumerate(files):
            filepath = os.path.join(data_dir, filename)
            print(f"  [{i+1}/{len(files)}] Attempting to load: {filename}")
            
            sat_instance, solution = load_dimacs_with_solution(filepath)
            
            if sat_instance is not None and solution is not None:
                training_data.append((sat_instance, solution))
                print(f"  [{i+1}/{len(files)}] ✅ {filename}")
            else:
                print(f"  [{i+1}/{len(files)}] ❌ {filename} (failed to load)")
    
    except Exception as e:
        print(f"Error loading training data: {e}")
        return []
    
    print(f"Successfully loaded {len(training_data)} training examples")
    return training_data


class SATFeatureExtractor:
    """Extract features from SAT instances for neural network input"""
    
    def __init__(self):
        self.feature_names = [
            'var_pos_freq', 'var_neg_freq', 'var_total_freq', 'var_balance',
            'num_vars', 'num_clauses', 'clause_var_ratio'
        ]
    
    def extract_features(self, sat_instance: SATInstance) -> List[float]:
        """
        Extract feature vector from SAT instance
        Args:
            sat_instance: The SAT problem
        Returns:
            Feature vector (flattened for neural network input)
        """
        features = []
        
        # Per-variable features
        var_pos_count = [0] * sat_instance.num_vars
        var_neg_count = [0] * sat_instance.num_vars
        
        # Count literal occurrences
        for clause in sat_instance.clauses:
            for literal in clause:
                if literal > 0:
                    var_pos_count[literal - 1] += 1
                else:
                    var_neg_count[abs(literal) - 1] += 1
        
        # Add per-variable features
        for i in range(sat_instance.num_vars):
            pos_freq = var_pos_count[i]
            neg_freq = var_neg_count[i]
            total_freq = pos_freq + neg_freq
            balance = (pos_freq - neg_freq) / max(total_freq, 1)  # Prevent division by zero
            
            features.extend([pos_freq, neg_freq, total_freq, balance])
        
        # Add global features
        features.extend([
            sat_instance.num_vars,
            len(sat_instance.clauses),
            len(sat_instance.clauses) / sat_instance.num_vars
        ])
        
        return features
    
    def get_feature_dim(self, num_vars: int) -> int:
        """Get the dimension of feature vector for given number of variables"""
        return num_vars * 4 + 3  # 4 features per variable + 3 global features


class SATSolverNetwork(nn.Module):
    """Neural Network that learns to solve SAT problems using MCMC layer"""
    
    def __init__(self, feature_dim: int, num_vars: int, hidden_dim: int = 128):
        super().__init__()
        self.num_vars = num_vars
        
        # Feature processing network
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_vars)  # Output logits for each variable
        )
        
        print(f"Created SAT Solver Network:")
        print(f"  Input features: {feature_dim}")
        print(f"  Hidden dimension: {hidden_dim}")
        print(f"  Output variables: {num_vars}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: features → logits
        Args:
            features: SAT instance features [batch_size, feature_dim]
        Returns:
            logits: Variable preferences [batch_size, num_vars]
        """
        logits = self.feature_net(features)
        return logits


class DifferentiableMCMCLayer:
    """
    Differentiable wrapper around MCMC layer for neural network training
    Implements the Fenchel-Young loss from the paper
    """
    
    def __init__(self, mcmc_layer: MCMCSATLayer):
        self.mcmc_layer = mcmc_layer
    
    def forward(self, logits: torch.Tensor, target_assignment: torch.Tensor = None, 
                num_iterations: int = 50) -> Tuple[torch.Tensor, float]:
        """
        Forward pass with gradient computation
        Args:
            logits: Learned preferences [num_vars]
            target_assignment: Ground truth assignment [num_vars] (optional)
            num_iterations: MCMC iterations
        Returns:
            (expectation, loss_value)
        """
        # Convert to Python lists for MCMC layer
        logits_list = logits.detach().numpy().tolist()
        
        # Run MCMC to get expectation
        expectation_list = self.mcmc_layer.forward(logits_list, num_iterations, burn_in=10)
        expectation = torch.tensor(expectation_list, dtype=torch.float32, requires_grad=True)
        
        # Compute loss if target provided
        loss_value = 0.0
        if target_assignment is not None:
            # Fenchel-Young loss: ||E[Y] - y*||^2
            target = target_assignment.float()
            loss_value = torch.nn.functional.mse_loss(expectation, target).item()
        
        return expectation, loss_value
    
    def compute_gradient_estimate(self, logits: torch.Tensor, target_assignment: torch.Tensor,
                                 num_iterations: int = 50) -> torch.Tensor:
        """
        Compute gradient estimate for the Fenchel-Young loss
        This is a simplified version - in practice would use the exact gradients from paper
        """
        logits_np = logits.detach().numpy().tolist()
        
        # Run MCMC to get expectation
        expectation_list = self.mcmc_layer.forward(logits_np, num_iterations, burn_in=10)
        expectation = torch.tensor(expectation_list, dtype=torch.float32)
        
        # Gradient of Fenchel-Young loss: ∇ω ϖ(ω;y) = E[Y] - y
        target = target_assignment.float()
        gradient = expectation - target
        
        return gradient


def train_sat_solver(training_data: List[Tuple[SATInstance, List[bool]]], 
                    num_epochs: int = 20, learning_rate: float = 0.001):
    """
    Train the SAT solver network
    Args:
        training_data: List of (sat_instance, solution) pairs
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    if not training_data:
        print("No training data available!")
        return None
    
    # Get dimensions from first example
    first_sat, first_solution = training_data[0]
    num_vars = first_sat.num_vars
    
    feature_extractor = SATFeatureExtractor()
    feature_dim = feature_extractor.get_feature_dim(num_vars)
    
    # Create network
    network = SATSolverNetwork(feature_dim, num_vars)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    
    print(f"\nTraining SAT Solver Network:")
    print(f"  Training examples: {len(training_data)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for i, (sat_instance, target_solution) in enumerate(training_data):
            # Extract features
            features = feature_extractor.extract_features(sat_instance)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Forward pass
            logits = network(features_tensor).squeeze(0)
            
            # Create MCMC layer for this instance
            neighborhood = BitFlipNeighborhood(hamming_distance=1)
            # mcmc_layer = MCMCSATLayer(sat_instance, neighborhood, temperature=1.5)
            mcmc_layer = MCMCSATLayer(sat_instance, neighborhood, temperature=2.0)
            diff_mcmc = DifferentiableMCMCLayer(mcmc_layer)
            
            # Compute gradient estimate (simplified training)
            target_tensor = torch.tensor([1.0 if x else 0.0 for x in target_solution], dtype=torch.float32)
            # gradient_estimate = diff_mcmc.compute_gradient_estimate(logits, target_tensor, num_iterations=30)
            gradient_estimate = diff_mcmc.compute_gradient_estimate(logits, target_tensor, num_iterations=30)
            
            # Manual gradient update (simplified)
            optimizer.zero_grad()
            
            # Create a dummy loss for backprop (this is a simplification)
            dummy_loss = torch.sum(logits * gradient_estimate)
            dummy_loss.backward()
            
            optimizer.step()
            
            total_loss += torch.norm(gradient_estimate).item()
        
        avg_loss = total_loss / len(training_data)
        print(f"  Epoch {epoch+1}/{num_epochs}: Average gradient norm = {avg_loss:.4f}")
    
    print("Training completed!")
    return network, feature_extractor


def test_learned_solver(network: SATSolverNetwork, feature_extractor: SATFeatureExtractor, 
                       test_file: str):
    """
    Test the learned SAT solver on new instances
    Args:
        network: Trained network
        feature_extractor: Feature extractor
        test_file: Path to test SAT file
    """
    print(f"\n=== Testing Learned SAT Solver on {test_file} ===")
    
    # Load test instance
    sat = load_dimacs_file(test_file)
    
    # Extract features and get logits
    features = feature_extractor.extract_features(sat)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        logits = network(features_tensor).squeeze(0)
    
    print(f"Learned logits: {[f'{x:.2f}' for x in logits.tolist()[:5]]}...")
    
    # Use learned logits with MCMC
    neighborhood = BitFlipNeighborhood(hamming_distance=1)
    mcmc_layer = MCMCSATLayer(sat, neighborhood, temperature=1.0, constraint_penalty=10.0)
    
    # Get expectation
    expectation = mcmc_layer.forward(logits.tolist(), num_iterations=100, burn_in=20)
    hard_assignment = [x > 0.5 for x in expectation]
    satisfaction = sat.evaluate(hard_assignment)
    
    print(f"Expectation-based assignment: {satisfaction}/{len(sat.clauses)} ({satisfaction/len(sat.clauses)*100:.1f}%)")
    
    # Also try to find best assignment
    best_assignment, best_score, best_satisfaction = mcmc_layer.get_best_assignment(
        logits.tolist(), num_iterations=100)
    
    print(f"Best assignment found: {best_satisfaction}/{len(sat.clauses)} ({best_satisfaction/len(sat.clauses)*100:.1f}%)")
    
    if best_satisfaction == len(sat.clauses):
        print("🎉 LEARNED SOLVER FOUND SOLUTION! 🎉")
        print(f"Solution: {best_assignment}")
    
    # Compare with random logits baseline
    random_logits = [random.gauss(0, 1) for _ in range(sat.num_vars)]
    random_best_assignment, _, random_best_satisfaction = mcmc_layer.get_best_assignment(
        random_logits, num_iterations=100)
    
    print(f"Random logits baseline: {random_best_satisfaction}/{len(sat.clauses)} ({random_best_satisfaction/len(sat.clauses)*100:.1f}%)")
    
    if best_satisfaction > random_best_satisfaction:
        print(f"✅ Learned solver better by {best_satisfaction - random_best_satisfaction} clauses!")
    elif best_satisfaction == random_best_satisfaction:
        print("➡️ Learned solver tied with random baseline")
    else:
        print(f"❌ Learned solver worse by {random_best_satisfaction - best_satisfaction} clauses")


def test_learned_solver_on_dataset(network: SATSolverNetwork, feature_extractor: SATFeatureExtractor, 
                                  test_dir: str, max_files: int = 50):
    """
    Test the learned SAT solver on a dataset of test instances
    Args:
        network: Trained network
        feature_extractor: Feature extractor
        test_dir: Directory containing test files
        max_files: Maximum number of test files to evaluate
    """
    print(f"\n=== Testing Learned SAT Solver on {test_dir}/ dataset ===")
    
    # Load test data
    test_data = load_training_data(test_dir, max_files=max_files)
    
    if not test_data:
        print("No test data available!")
        return
    
    total_learned_satisfaction = 0
    total_random_satisfaction = 0
    total_clauses = 0
    solutions_found = 0
    random_solutions_found = 0
    
    for i, (sat_instance, true_solution) in enumerate(test_data):
        print(f"\n--- Test {i+1}/{len(test_data)} ---")
        
        # Extract features and get learned logits
        features = feature_extractor.extract_features(sat_instance)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = network(features_tensor).squeeze(0)
        
        # Test learned logits with MCMC
        neighborhood = BitFlipNeighborhood(hamming_distance=1)
        mcmc_layer = MCMCSATLayer(sat_instance, neighborhood, temperature=2.0, constraint_penalty=10.0)
        
        learned_best_assignment, _, learned_satisfaction = mcmc_layer.get_best_assignment(
            logits.tolist(), num_iterations=200)
        
        # Compare with random logits baseline
        random_logits = [random.gauss(0, 1) for _ in range(sat_instance.num_vars)]
        random_best_assignment, _, random_satisfaction = mcmc_layer.get_best_assignment(
            random_logits, num_iterations=200)
        
        # Accumulate statistics
        total_learned_satisfaction += learned_satisfaction
        total_random_satisfaction += random_satisfaction
        total_clauses += len(sat_instance.clauses)
        
        if learned_satisfaction == len(sat_instance.clauses):
            solutions_found += 1
            print(f"  🎉 SOLUTION FOUND by learned solver!")

        if random_satisfaction == len(sat_instance.clauses):
            random_solutions_found += 1
        
        print(f"  Learned: {learned_satisfaction}/{len(sat_instance.clauses)} ({learned_satisfaction/len(sat_instance.clauses)*100:.1f}%)")
        print(f"  Random:  {random_satisfaction}/{len(sat_instance.clauses)} ({random_satisfaction/len(sat_instance.clauses)*100:.1f}%)")
        print(f"  Difference: {learned_satisfaction - random_satisfaction:+d} clauses")
    
    # Final statistics
    print(f"\n=== Overall Results ===")
    print(f"Test instances: {len(test_data)}")
    print(f"Complete solutions found: {solutions_found}/{len(test_data)} ({solutions_found/len(test_data)*100:.1f}%)")
    print(f"Average learned satisfaction: {total_learned_satisfaction/total_clauses*100:.1f}%")
    print(f"Average random satisfaction:  {total_random_satisfaction/total_clauses*100:.1f}%")
    print(f"Random complete solutions: {random_solutions_found}/{len(test_data)} ({random_solutions_found/len(test_data)*100:.1f}%)")
    print(f"Improvement: {(total_learned_satisfaction - total_random_satisfaction)/total_clauses*100:.2f} percentage points")
    
    if total_learned_satisfaction > total_random_satisfaction:
        print("✅ Learned solver outperforms random baseline!")
    elif total_learned_satisfaction == total_random_satisfaction:
        print("➡️ Learned solver ties with random baseline")
    else:
        print("❌ Learned solver underperforms random baseline")

def test_real_sat_file(filename: str):
    """Test loading and analyzing a real SAT file"""
    print(f"=== Testing Real SAT File: {filename} ===")
    
    # Load the file
    sat = load_dimacs_file(filename)
    sat.print_summary()
    
    # Test with some random assignments
    print(f"\n=== Testing Random Assignments ===")
    neighborhood = BitFlipNeighborhood(hamming_distance=1)
    
    for trial in range(3):
        # Generate random assignment
        assignment = [random.choice([True, False]) for _ in range(sat.num_vars)]
        satisfied = sat.evaluate(assignment)
        is_solution = sat.is_satisfiable(assignment)
        
        print(f"\nTrial {trial + 1}:")
        print(f"  Assignment: {assignment}")
        print(f"  Satisfied: {satisfied}/{len(sat.clauses)} clauses ({satisfied/len(sat.clauses)*100:.1f}%)")
        print(f"  Is solution: {is_solution}")
        
        # Check a few neighbors
        neighbors = neighborhood.get_neighbors(assignment)
        print(f"  Checking first 3 of {len(neighbors)} neighbors:")
        
        for i, neighbor in enumerate(neighbors[:3]):
            neighbor_satisfied = sat.evaluate(neighbor)
            neighbor_is_solution = sat.is_satisfiable(neighbor)
            improvement = neighbor_satisfied - satisfied
            print(f"    Neighbor {i+1}: {neighbor_satisfied}/{len(sat.clauses)} " +
                  f"({improvement:+d})" + (" SOLUTION!" if neighbor_is_solution else ""))


def test_mcmc_layer(filename: str):
    """Test the MCMC layer on a real SAT file"""
    print(f"=== Testing MCMC Layer: {filename} ===")
    
    # Load the SAT instance
    sat = load_dimacs_file(filename)
    neighborhood = BitFlipNeighborhood(hamming_distance=1)
    mcmc_layer = MCMCSATLayer(sat, neighborhood, temperature=2.0, constraint_penalty=5.0)
    
    # Test with different logit strategies
    strategies = [
        ("Random logits", [random.gauss(0, 1) for _ in range(sat.num_vars)]),
        ("Zero logits", [0.0] * sat.num_vars),
        ("Positive bias", [1.0] * sat.num_vars),
        ("Negative bias", [-1.0] * sat.num_vars),
    ]
    
    for name, logits in strategies:
        print(f"\n--- Testing {name} ---")
        print(f"Logits: {[f'{x:.2f}' for x in logits[:5]]}...")
        
        # Get expectation (main layer output)
        expectation = mcmc_layer.forward(logits, num_iterations=50, burn_in=10)
        print(f"Expectation: {[f'{x:.3f}' for x in expectation[:5]]}...")
        
        # Convert expectation to hard assignment (threshold at 0.5)
        hard_assignment = [x > 0.5 for x in expectation]
        satisfaction = sat.evaluate(hard_assignment)
        print(f"Hard assignment satisfaction: {satisfaction}/{len(sat.clauses)} ({satisfaction/len(sat.clauses)*100:.1f}%)")
        
        # Find best assignment encountered
        best_assignment, best_score, best_satisfaction = mcmc_layer.get_best_assignment(logits, num_iterations=50)
        print(f"Best assignment found: {best_satisfaction}/{len(sat.clauses)} ({best_satisfaction/len(sat.clauses)*100:.1f}%)")
        
        if best_satisfaction == len(sat.clauses):
            print("🎉 SOLUTION FOUND! 🎉")
            print(f"Solution: {best_assignment}")


if __name__ == "__main__":
    print("3-SAT Foundation + MCMC + Neural Network")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        
        # Run basic tests first
        test_real_sat_file(filename)
        print("\n" + "=" * 50)
        test_mcmc_layer(filename)
        
        # Neural network training and testing
        print("\n" + "=" * 50)
        print("=== Neural Network Training Phase ===")
        
        # Load training data from directory
        training_data = load_training_data("traindata", max_files=800)  # Use subset for faster training
        
        if training_data:
            # Train the network
            network, feature_extractor = train_sat_solver(training_data, num_epochs=20, learning_rate=0.005)
            
            # Test on individual file first
            test_learned_solver(network, feature_extractor, filename)
            
            # Test on dataset
            test_learned_solver_on_dataset(network, feature_extractor, "testdata", max_files=200)
        else:
            print("Could not load training data from traindata/ directory")
        
    else:
        # No file specified - check if we can do full training/testing
        print("=== Checking for training data directories ===")
        
        if os.path.exists("traindata") and os.path.exists("testdata"):
            print("Found traindata/ and testdata/ directories!")
            print("Loading training data...")
            
            # Load training data
            training_data = load_training_data("traindata", max_files=800)
            
            if training_data:
                # Train the network
                network, feature_extractor = train_sat_solver(training_data, num_epochs=20, learning_rate=0.005)
                
                # Test on dataset
                test_learned_solver_on_dataset(network, feature_extractor, "testdata", max_files=200)
            else:
                print("Failed to load training data")
        else:
            print("traindata/ and/or testdata/ directories not found")
            print("Running built-in tests instead...")
            
            # Run built-in tests - create sample instance
            def test_sat_instance():
                print("=== Testing Sample SAT Instance ===")
                clauses = [(1, 2, 3), (-1, -2, 3)]
                sat = SATInstance(num_vars=3, clauses=clauses)
                sat.print_summary()
                
                test_assignments = [
                    [True, True, True],
                    [True, True, False],
                    [False, False, False],
                    [False, False, True],
                ]
                
                for i, assignment in enumerate(test_assignments):
                    satisfied = sat.evaluate(assignment)
                    is_solution = sat.is_satisfiable(assignment)
                    print(f"Assignment {i+1}: {assignment} -> {satisfied}/{len(sat.clauses)} ({'SOLUTION' if is_solution else 'partial'})")
            
            test_sat_instance()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
