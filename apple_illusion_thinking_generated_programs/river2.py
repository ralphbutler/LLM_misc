#!/usr/bin/env python3

import sys
from collections import deque
from itertools import combinations

class RiverCrossingOptimized:
    def __init__(self, n, boat_capacity=None):
        self.n = n
        if boat_capacity is None:
            self.boat_capacity = 3 if n > 3 else 2
        else:
            self.boat_capacity = boat_capacity
        self.total_people = 2 * n
        
        # Use bit representation: bit i set means person i is on right bank
        # People indexed as: a1=0, a2=1, ..., aN=N-1, A1=N, A2=N+1, ..., AN=2N-1
        self.goal_state = (2**self.total_people - 1, 'right')  # All on right bank
        self.initial_state = (0, 'left')  # All on left bank
    
    def get_people_names(self):
        """Get mapping from bit index to person name."""
        names = []
        for i in range(self.n):
            names.append(f'a{i+1}')  # actors
        for i in range(self.n):
            names.append(f'A{i+1}')  # agents
        return names
    
    def is_safe_state(self, state_bits):
        """Check if a state satisfies the safety constraint using bit operations."""
        left_bank = ~state_bits & ((1 << self.total_people) - 1)
        right_bank = state_bits
        
        return self._check_bank_safety(left_bank) and self._check_bank_safety(right_bank)
    
    def _check_bank_safety(self, bank_bits):
        """Check safety for one bank using bit operations."""
        actors_bits = bank_bits & ((1 << self.n) - 1)  # First n bits
        agents_bits = (bank_bits >> self.n) & ((1 << self.n) - 1)  # Next n bits
        
        # Check each actor
        for i in range(self.n):
            actor_present = (actors_bits >> i) & 1
            if actor_present:
                agent_present = (agents_bits >> i) & 1
                # Actor is unsafe if their agent is absent but other agents are present
                if not agent_present and agents_bits != 0:
                    return False
        return True
    
    def get_possible_moves(self, state_bits, boat_side):
        """Generate possible moves using bit operations."""
        if boat_side == 'left':
            available_bits = ~state_bits & ((1 << self.total_people) - 1)
            new_boat_side = 'right'
        else:
            available_bits = state_bits
            new_boat_side = 'left'
        
        available_people = []
        for i in range(self.total_people):
            if (available_bits >> i) & 1:
                available_people.append(i)
        
        moves = []
        
        # Try all combinations of 1 to boat_capacity people
        for size in range(1, min(self.boat_capacity + 1, len(available_people) + 1)):
            for combo in combinations(available_people, size):
                # Calculate new state
                move_bits = 0
                for person in combo:
                    move_bits |= (1 << person)
                
                if boat_side == 'left':
                    new_state_bits = state_bits | move_bits
                else:
                    new_state_bits = state_bits & ~move_bits
                
                # Check if new state is safe
                if self.is_safe_state(new_state_bits):
                    moves.append((combo, (new_state_bits, new_boat_side)))
        
        return moves
    
    def solve(self):
        """Solve using bidirectional BFS for better performance."""
        if not self.is_safe_state(self.initial_state[0]):
            return None
        
        # Forward search from initial state
        forward_queue = deque([(self.initial_state, [])])
        forward_visited = {self.initial_state: []}
        
        # Backward search from goal state  
        backward_queue = deque([(self.goal_state, [])])
        backward_visited = {self.goal_state: []}
        
        while forward_queue or backward_queue:
            # Expand forward search
            if forward_queue:
                current_state, path = forward_queue.popleft()
                state_bits, boat_side = current_state
                
                # Check if we've met the backward search
                if current_state in backward_visited:
                    return path + list(reversed(backward_visited[current_state]))
                
                for move, next_state in self.get_possible_moves(state_bits, boat_side):
                    if next_state not in forward_visited:
                        new_path = path + [move]
                        forward_visited[next_state] = new_path
                        forward_queue.append((next_state, new_path))
            
            # Expand backward search
            if backward_queue:
                current_state, path = backward_queue.popleft()
                state_bits, boat_side = current_state
                
                # Check if we've met the forward search
                if current_state in forward_visited:
                    return forward_visited[current_state] + list(reversed(path))
                
                # For backward search, reverse the boat direction
                reverse_boat_side = 'left' if boat_side == 'right' else 'right'
                
                for move, next_state in self.get_possible_moves(state_bits, reverse_boat_side):
                    # Reverse the state for backward search
                    reverse_next_state = (next_state[0], boat_side)
                    
                    if reverse_next_state not in backward_visited:
                        new_path = [move] + path
                        backward_visited[reverse_next_state] = new_path
                        backward_queue.append((reverse_next_state, new_path))
        
        return None
    
    def format_solution(self, solution):
        """Format the solution for output."""
        if solution is None:
            return "No solution found!"
        
        names = self.get_people_names()
        formatted_moves = []
        
        for i, move in enumerate(solution, 1):
            people = [names[person_idx] for person_idx in sorted(move)]
            people_str = ', '.join(people)
            formatted_moves.append(f"Move {i}: {people_str} cross the river")
        
        return formatted_moves

def test_boat_capacities(n, max_capacity=None):
    """Test different boat capacities to find minimum that works."""
    if max_capacity is None:
        max_capacity = n + 2
    
    for capacity in range(2, max_capacity + 1):
        print(f"Testing N={n} with boat capacity {capacity}...")
        solver = RiverCrossingOptimized(n, capacity)
        solution = solver.solve()
        if solution:
            print(f"✓ Solution found with capacity {capacity} in {len(solution)} moves")
            return capacity
        else:
            print(f"✗ No solution with capacity {capacity}")
    
    print(f"No solution found for N={n} with capacities 2-{max_capacity}")
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python river2.py <N> [boat_capacity|test]")
        sys.exit(1)
    
    try:
        n = int(sys.argv[1])
        if n <= 0:
            raise ValueError("N must be positive")
        if n > 15:
            print("Warning: N > 15 may take a very long time or run out of memory")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if len(sys.argv) > 2:
        if sys.argv[2] == "test":
            test_boat_capacities(n)
            return
        else:
            try:
                boat_capacity = int(sys.argv[2])
            except ValueError:
                print("Error: boat_capacity must be an integer or 'test'")
                sys.exit(1)
    else:
        boat_capacity = None
    
    solver = RiverCrossingOptimized(n, boat_capacity)
    
    print(f"Solving river crossing problem for N = {n}")
    print(f"Boat capacity: {solver.boat_capacity}")
    print()
    
    solution = solver.solve()
    
    if solution is None:
        print("No solution found!")
    else:
        print(f"Solution found in {len(solution)} moves:")
        print()
        formatted_moves = solver.format_solution(solution)
        for move in formatted_moves:
            print(move)

if __name__ == "__main__":
    main()