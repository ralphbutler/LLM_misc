#!/usr/bin/env python3
import sys
from typing import List, Tuple, Optional

def solve_checker_jumping(n: int) -> List[Tuple[int, int]]:
    """
    Solve the checker jumping puzzle for n checkers per side.
    Returns a list of moves as (from_pos, to_pos) tuples.
    """
    # Initialize board: 1s (red) on left, _ in middle, 2s (blue) on right
    board = [1] * n + [0] + [2] * n
    target = [2] * n + [0] + [1] * n
    
    moves = []
    
    def find_empty() -> int:
        return board.index(0)
    
    def is_valid_move(from_pos: int, to_pos: int) -> bool:
        if to_pos != find_empty():
            return False
        
        checker = board[from_pos]
        if checker == 0:
            return False
        
        # Red checkers (1) can only move right, blue checkers (2) can only move left
        if checker == 1 and to_pos <= from_pos:
            return False
        if checker == 2 and to_pos >= from_pos:
            return False
        
        distance = abs(to_pos - from_pos)
        
        # Slide move (distance 1)
        if distance == 1:
            return True
        
        # Jump move (distance 2, must jump over opposite color)
        if distance == 2:
            middle_pos = (from_pos + to_pos) // 2
            middle_checker = board[middle_pos]
            # Must jump over opposite color
            return middle_checker != 0 and middle_checker != checker
        
        return False
    
    def make_move(from_pos: int, to_pos: int):
        board[to_pos] = board[from_pos]
        board[from_pos] = 0
        moves.append((from_pos, to_pos))
    
    def solve_recursive() -> bool:
        if board == target:
            return True
        
        empty_pos = find_empty()
        
        # Try all possible moves
        for pos in range(len(board)):
            if board[pos] != 0:
                # Try slide move
                if is_valid_move(pos, empty_pos):
                    old_board = board[:]
                    make_move(pos, empty_pos)
                    if solve_recursive():
                        return True
                    # Backtrack
                    board[:] = old_board
                    moves.pop()
        
        return False
    
    if solve_recursive():
        return moves
    else:
        return []

def print_board(board: List[int]):
    """Print the board state with proper formatting."""
    display = []
    for cell in board:
        if cell == 0:
            display.append('_')
        elif cell == 1:
            display.append('R')
        else:
            display.append('B')
    print(' '.join(display))

def main():
    if len(sys.argv) != 2:
        print("Usage: python checker_jumping.py <N>")
        print("Where N is the number of checkers per side")
        sys.exit(1)
    
    try:
        n = int(sys.argv[1])
        if n <= 0:
            raise ValueError("N must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Solving checker jumping puzzle for N={n}")
    
    # Initialize and display starting board
    board = [1] * n + [0] + [2] * n
    print("\nStarting position:")
    print_board(board)
    
    # Solve puzzle
    moves = solve_checker_jumping(n)
    
    if not moves:
        print("No solution found!")
        return
    
    print(f"\nSolution found in {len(moves)} moves:")
    
    # Apply and display each move
    for i, (from_pos, to_pos) in enumerate(moves):
        checker = board[from_pos]
        board[to_pos] = checker
        board[from_pos] = 0
        
        checker_type = 'R' if checker == 1 else 'B'
        print(f"Move {i+1}: {checker_type} from position {from_pos} to position {to_pos}")
        print_board(board)
    
    print(f"\nPuzzle solved in {len(moves)} moves!")

if __name__ == "__main__":
    main()