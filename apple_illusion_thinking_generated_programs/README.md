# Apple Paper Programs by Claude

This repository contains Python implementations of four classic constraint satisfaction and planning puzzles, based on the Apple paper on LLM reasoning abilities. These programs demonstrate algorithmic solutions to problems that test sequential reasoning and planning capabilities.

## Programs Overview

### 1. Blocks Puzzle (`blocks1.py`)
A constraint satisfaction puzzle where blocks must be rearranged between stacks in a specific alternating pattern.

**Usage:**
```bash
python blocks1.py 1 2 3 4 5 0 6 7 8 9
```

- Input format: Space-separated numbers where `0` separates left and middle stacks
- Example: `1 2 3 4 5 0 6 7 8 9` creates left stack [1,2,3,4,5], middle stack [6,7,8,9], empty right stack
- The program finds the minimum moves to achieve the goal configuration using alternating pattern
- Uses BFS to find optimal solution

### 2. Checker Jumping (`checkers1.py`)
A one-dimensional puzzle where red and blue checkers must swap positions following movement rules.

**Usage:**
```bash
python checkers1.py <N>
```

- `N` is the number of checkers per side
- Initial: N red checkers (R), empty space, N blue checkers (B)
- Goal: N blue checkers, empty space, N red checkers
- Movement rules: Slide to adjacent empty space or jump over opposite color
- Red checkers only move right, blue checkers only move left

**Examples:**
- `python checkers1.py 3` - Simple 3v3 game
- `python checkers1.py 20` - Larger puzzle (440 moves, ~44 seconds)

### 3. Towers of Hanoi (`hanoi1.py`)
Classic recursive puzzle to move all disks from one peg to another following size constraints.

**Usage:**
```bash
python hanoi1.py <number_of_disks>
```

- Moves all disks from Tower A to Tower C using Tower B as auxiliary
- Rules: Move one disk at a time, only top disk, never place larger on smaller
- Shows visual representation and step-by-step moves for ≤4 disks
- For larger numbers, shows only start/end states and move count
- Total moves required: 2^n - 1

**Examples:**
- `python hanoi1.py 3` - Shows all 7 moves step-by-step
- `python hanoi1.py 5` - Shows condensed output (31 moves total)

### 4. River Crossing (`river1.py` and `river2.py`)
Multi-agent coordination puzzle where actors and their agents must cross a river safely.

**Basic Version (`river1.py`):**
```bash
python river1.py <N>
```

**Optimized Version (`river2.py`):**
```bash
python river2.py <N> [boat_capacity|test]
python river2.py <N> test  # Test different boat capacities
```

- `N` actors (a1, a2, ..., aN) and N agents (A1, A2, ..., AN) must cross river
- Boat capacity: 2 for N≤3, 3 for N>3
- Safety constraint: Actor cannot be with other agents unless their own agent is present
- `river2.py` includes bidirectional BFS optimization and boat capacity testing

**Working Solutions:**
- N=1 to N=5: Solutions exist
- N≥6: No solutions exist with standard boat capacities due to safety constraints

## Technical Notes

- The programs demonstrate code-writing ability rather than raw reasoning ability
- River crossing requires tuning both N (travelers) and K (boat capacity) parameters  
- Blocks puzzle uses simplified assumptions about stack count and configurations
- Performance varies significantly: checker jumping with N=20 takes ~44 seconds on M4 MacBook Pro

## Requirements
- Python 3.x
- No external dependencies (uses only standard library)

## Files Description
- `*.py` - Executable Python programs
- `*_notes.txt` - Development notes and problem descriptions
- `blocks_prompt.txt` - Original problem specification for blocks puzzle