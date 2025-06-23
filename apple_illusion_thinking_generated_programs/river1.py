#!/usr/bin/env python3

import sys
from collections import deque
from itertools import combinations

def is_safe_state(left_bank, right_bank):
    """Check if a state satisfies the safety constraint."""
    def check_bank_safety(bank):
        actors = set()
        agents = set()
        
        for person in bank:
            if person.startswith('a'):
                actors.add(int(person[1:]))
            else:  # person.startswith('A')
                agents.add(int(person[1:]))
        
        # For each actor, check if they're safe
        for actor in actors:
            # Actor is safe if their agent is present, or no other agents are present
            if actor not in agents and len(agents) > 0:
                return False
        return True
    
    return check_bank_safety(left_bank) and check_bank_safety(right_bank)

def get_possible_moves(state, boat_capacity):
    """Generate all possible moves from current state."""
    left_bank, right_bank, boat_side = state
    
    if boat_side == 'left':
        available = left_bank
        destination = right_bank
        new_boat_side = 'right'
    else:
        available = right_bank
        destination = left_bank
        new_boat_side = 'left'
    
    moves = []
    
    # Generate all possible combinations of people to move (1 to boat_capacity)
    for size in range(1, min(boat_capacity + 1, len(available) + 1)):
        for combo in combinations(available, size):
            new_available = available - set(combo)
            new_destination = destination | set(combo)
            
            if boat_side == 'left':
                new_state = (frozenset(new_available), frozenset(new_destination), new_boat_side)
            else:
                new_state = (frozenset(new_destination), frozenset(new_available), new_boat_side)
            
            # Check if the new state is safe
            if is_safe_state(new_state[0], new_state[1]):
                moves.append((combo, new_state))
    
    return moves

def solve_river_crossing(n):
    """Solve the river crossing problem using BFS."""
    # Determine boat capacity
    boat_capacity = 2 if n <= 3 else 3
    
    # Initialize state: (left_bank, right_bank, boat_side)
    # People are represented as 'a1', 'a2', ..., 'aN', 'A1', 'A2', ..., 'AN'
    initial_people = set()
    for i in range(1, n + 1):
        initial_people.add(f'a{i}')  # actors
        initial_people.add(f'A{i}')  # agents
    
    initial_state = (frozenset(initial_people), frozenset(), 'left')
    goal_state = (frozenset(), frozenset(initial_people), 'right')
    
    if not is_safe_state(initial_state[0], initial_state[1]):
        return None  # Initial state is not safe
    
    # BFS
    queue = deque([(initial_state, [])])
    visited = {initial_state}
    
    while queue:
        current_state, path = queue.popleft()
        
        if current_state == goal_state:
            return path
        
        for move, next_state in get_possible_moves(current_state, boat_capacity):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [move]))
    
    return None  # No solution found

def format_move(move, move_num):
    """Format a move for output."""
    people = sorted(list(move))
    people_str = ', '.join(people)
    return f"Move {move_num}: {people_str} cross the river"

def main():
    if len(sys.argv) != 2:
        print("Usage: python river_crossing.py <N>")
        sys.exit(1)
    
    try:
        n = int(sys.argv[1])
        if n <= 0:
            raise ValueError("N must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Solving river crossing problem for N = {n}")
    print(f"Boat capacity: {2 if n <= 3 else 3}")
    print()
    
    solution = solve_river_crossing(n)
    
    if solution is None:
        print("No solution found!")
    else:
        print(f"Solution found in {len(solution)} moves:")
        print()
        for i, move in enumerate(solution, 1):
            print(format_move(move, i))

if __name__ == "__main__":
    main()