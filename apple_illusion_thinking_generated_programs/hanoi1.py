#!/usr/bin/env python3
import sys

class TowersOfHanoi:
    def __init__(self, n):
        self.n = n
        self.towers = [list(range(n, 0, -1)), [], []]
        self.move_count = 0
        self.show_steps = n <= 4  # Show all steps for small numbers of disks
        
    def print_state(self, message=""):
        if message:
            print(f"\n{message}")
        
        # Find the height we need to display
        max_height = max(len(tower) for tower in self.towers)
        
        # Print from top to bottom
        for level in range(max_height - 1, -1, -1):
            line = ""
            for i, tower in enumerate(self.towers):
                if level < len(tower):
                    disk = tower[level]
                    # Create visual representation of disk
                    padding = " " * (self.n - disk)
                    disk_repr = "*" * (disk * 2 + 1)
                    line += f"{padding}{disk_repr}{padding}"
                else:
                    # Empty space
                    line += " " * (self.n * 2 + 1)
                
                if i < 2:  # Add separator between towers
                    line += " | "
            print(line)
        
        # Print base
        base_line = ""
        for i in range(3):
            base_line += "-" * (self.n * 2 + 1)
            if i < 2:
                base_line += " | "
        print(base_line)
        
        # Print tower labels
        label_line = ""
        for i in range(3):
            label = f"Tower {chr(65 + i)}"
            padding = " " * ((self.n * 2 + 1 - len(label)) // 2)
            label_line += f"{padding}{label}{padding}"
            if len(label) % 2 == 0 and self.n * 2 + 1 > len(label):
                label_line += " "
            if i < 2:
                label_line += " | "
        print(label_line)
        print()
    
    def move_disk(self, from_tower, to_tower):
        disk = self.towers[from_tower].pop()
        self.towers[to_tower].append(disk)
        self.move_count += 1
        
        if self.show_steps:
            tower_names = ['A', 'B', 'C']
            print(f"Move {self.move_count}: Move disk {disk} from Tower {tower_names[from_tower]} to Tower {tower_names[to_tower]}")
            self.print_state()
    
    def solve(self, n, from_tower, to_tower, aux_tower):
        if n == 1:
            self.move_disk(from_tower, to_tower)
        else:
            # Move n-1 disks from source to auxiliary tower
            self.solve(n - 1, from_tower, aux_tower, to_tower)
            # Move the largest disk from source to destination
            self.move_disk(from_tower, to_tower)
            # Move n-1 disks from auxiliary to destination
            self.solve(n - 1, aux_tower, to_tower, from_tower)
    
    def run(self):
        print(f"Towers of Hanoi with {self.n} disks")
        print("=" * 50)
        
        self.print_state("Initial state:")
        
        if not self.show_steps:
            print(f"Solving... (showing only start and end states for {self.n} disks)")
            print()
        
        self.solve(self.n, 0, 2, 1)  # Move from Tower A to Tower C using Tower B
        
        self.print_state("Final state:")
        print(f"Total moves required: {self.move_count}")
        print(f"Minimum possible moves: {2**self.n - 1}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python hanoi.py <number_of_disks>")
        print("Example: python hanoi.py 3")
        sys.exit(1)
    
    try:
        n = int(sys.argv[1])
        if n <= 0:
            raise ValueError("Number of disks must be positive")
        if n > 20:
            print("Warning: Large numbers of disks will require many moves!")
            print(f"Number of moves for {n} disks: {2**n - 1}")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please provide a positive integer for the number of disks.")
        sys.exit(1)
    
    hanoi = TowersOfHanoi(n)
    hanoi.run()

if __name__ == "__main__":
    main()