from collections import deque
from typing import List, Tuple, Optional
import sys

def parse_input(input_str: str) -> List[List[int]]:
    """Parse input string into initial stack configuration."""
    numbers = list(map(int, input_str.split()))
    zero_index = numbers.index(0)
    
    left_stack = numbers[:zero_index]
    middle_stack = numbers[zero_index + 1:]
    right_stack = []
    
    return [left_stack, middle_stack, right_stack]

def generate_goal_state(left_stack: List[int], middle_stack: List[int]) -> List[List[int]]:
    """Generate goal state using alternating pattern."""
    left_copy = left_stack.copy()
    middle_copy = middle_stack.copy()
    goal_stack = []
    
    from_middle = True
    while left_copy and middle_copy:
        if from_middle:
            goal_stack.append(middle_copy.pop())
        else:
            goal_stack.append(left_copy.pop())
        from_middle = not from_middle
    
    while left_copy:
        goal_stack.append(left_copy.pop())
    
    while middle_copy:
        goal_stack.append(middle_copy.pop())
    
    return [goal_stack, [], []]

def get_possible_moves(stacks: List[List[int]]) -> List[Tuple[int, int]]:
    """Get all possible moves from current state."""
    moves = []
    
    for from_stack in range(3):
        if stacks[from_stack]:
            for to_stack in range(3):
                if from_stack != to_stack:
                    moves.append((from_stack, to_stack))
    
    return moves

def apply_move(stacks: List[List[int]], move: Tuple[int, int]) -> List[List[int]]:
    """Apply a move and return new state."""
    from_stack, to_stack = move
    new_stacks = [stack.copy() for stack in stacks]
    
    if new_stacks[from_stack]:
        block = new_stacks[from_stack].pop()
        new_stacks[to_stack].append(block)
    
    return new_stacks

def state_to_tuple(stacks: List[List[int]]) -> Tuple:
    """Convert state to hashable tuple."""
    return tuple(tuple(stack) for stack in stacks)

def solve_blocks_puzzle(input_str: str) -> Optional[List[Tuple[int, int]]]:
    """Solve the blocks puzzle using BFS."""
    initial_stacks = parse_input(input_str)
    goal_stacks = generate_goal_state(initial_stacks[0], initial_stacks[1])
    
    initial_state = state_to_tuple(initial_stacks)
    goal_state = state_to_tuple(goal_stacks)
    
    if initial_state == goal_state:
        return []
    
    queue = deque([(initial_stacks, [])])
    visited = {initial_state}
    
    while queue:
        current_stacks, moves = queue.popleft()
        
        for move in get_possible_moves(current_stacks):
            new_stacks = apply_move(current_stacks, move)
            new_state = state_to_tuple(new_stacks)
            
            if new_state == goal_state:
                return moves + [move]
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_stacks, moves + [move]))
    
    return None

def print_solution(input_str: str):
    """Print the solution for the blocks puzzle."""
    initial_stacks = parse_input(input_str)
    goal_stacks = generate_goal_state(initial_stacks[0], initial_stacks[1])
    
    print(f"Initial state: {initial_stacks}")
    print(f"Goal state: {goal_stacks}")
    
    solution = solve_blocks_puzzle(input_str)
    
    if solution is None:
        print("No solution found!")
    elif solution == []:
        print("Already at goal state!")
    else:
        print(f"Solution found in {len(solution)} moves:")
        for i, (from_stack, to_stack) in enumerate(solution, 1):
            print(f"Move {i}: Stack {from_stack} -> Stack {to_stack}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python blocks_puzzle.py <numbers>")
        print("Example: python blocks_puzzle.py 1 2 3 4 5 0 6 7 8 9")
        sys.exit(1)
    
    input_str = " ".join(sys.argv[1:])
    print_solution(input_str)