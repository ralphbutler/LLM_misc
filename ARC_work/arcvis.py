import pygame
import json
import sys

# Initialize Pygame
pygame.init()

# Constants
CELL_SIZE = 60
MIN_CELL_SIZE = 20
GRID_LINE = 1
PADDING = 20
LABEL_HEIGHT = 30
MAX_WINDOW_WIDTH = 1200
MAX_WINDOW_HEIGHT = 800
SCROLL_SPEED = 20
SCROLL_BAR_WIDTH = 15
BACKGROUND_COLOR = (200, 200, 200)

# Colors for different numbers (0-9)
COLORS = {
    0: (0, 0, 0),       # Black for empty space
    1: (255, 0, 0),     # Red
    2: (0, 255, 0),     # Green
    3: (0, 0, 255),     # Blue
    4: (255, 255, 0),   # Yellow
    5: (255, 0, 255),   # Magenta
    6: (0, 255, 255),   # Cyan
    7: (255, 128, 0),   # Orange
    8: (128, 0, 255),   # Purple
    9: (0, 128, 0)      # Dark Green
}

class ScrollableWindow:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.scroll_y = 0
        self.cell_size = CELL_SIZE
        self.resize_display()
        
    def resize_display(self):
        # Calculate content dimensions
        max_pairs = max(len(self.puzzle['train']), len(self.puzzle['test']))
        self.max_grid_height = max(
            max(len(pair['input']) for pair in self.puzzle['train'] + self.puzzle['test']),
            max(len(pair['output']) for pair in self.puzzle['train'] + self.puzzle['test'])
        )
        self.max_grid_width = max(
            max(len(row) for pair in self.puzzle['train'] + self.puzzle['test'] for row in pair['input']),
            max(len(row) for pair in self.puzzle['train'] + self.puzzle['test'] for row in pair['output'])
        )
        
        # Calculate initial content size
        content_width = (2 * self.max_grid_width * self.cell_size) + (3 * PADDING)
        self.content_height = ((max_pairs + 1) * self.max_grid_height * self.cell_size) + \
                            ((max_pairs + 2) * PADDING) + ((max_pairs + 1) * LABEL_HEIGHT)
        
        # Add scroll bar width if content is taller than window
        if self.content_height > MAX_WINDOW_HEIGHT:
            content_width += SCROLL_BAR_WIDTH
            
        # Scale if needed
        if content_width > MAX_WINDOW_WIDTH or self.content_height > MAX_WINDOW_HEIGHT:
            scale = min(
                (MAX_WINDOW_WIDTH - SCROLL_BAR_WIDTH) / content_width,
                MAX_WINDOW_HEIGHT / self.content_height
            )
            new_cell_size = max(int(self.cell_size * scale), MIN_CELL_SIZE)
            if new_cell_size != self.cell_size:
                self.cell_size = new_cell_size
                # Recalculate content size with new cell size
                self.content_width = (2 * self.max_grid_width * self.cell_size) + (3 * PADDING)
                self.content_height = ((max_pairs + 1) * self.max_grid_height * self.cell_size) + \
                                   ((max_pairs + 2) * PADDING) + ((max_pairs + 1) * LABEL_HEIGHT)
        
        # Set window size
        self.window_width = min(content_width, MAX_WINDOW_WIDTH)
        self.window_height = min(self.content_height, MAX_WINDOW_HEIGHT)
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEWHEEL:
            self.scroll_y = max(
                min(self.scroll_y - event.y * SCROLL_SPEED, 
                    self.content_height - self.window_height),
                0
            )
        elif event.type == pygame.VIDEORESIZE:
            global MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT
            MAX_WINDOW_WIDTH = event.w
            MAX_WINDOW_HEIGHT = event.h
            self.resize_display()
            
    def draw_scrollbar(self):
        if self.content_height > self.window_height:
            # Calculate scrollbar height and position
            scrollbar_height = max(
                self.window_height * (self.window_height / self.content_height),
                20
            )
            scrollbar_pos = (self.scroll_y / self.content_height) * self.window_height
            
            # Draw scrollbar background
            pygame.draw.rect(self.screen, (128, 128, 128),
                           (self.window_width - SCROLL_BAR_WIDTH, 0,
                            SCROLL_BAR_WIDTH, self.window_height))
            
            # Draw scrollbar
            pygame.draw.rect(self.screen, (64, 64, 64),
                           (self.window_width - SCROLL_BAR_WIDTH, scrollbar_pos,
                            SCROLL_BAR_WIDTH, scrollbar_height))

    def draw_grid(self, grid, x_offset, y_offset):
        """Draw a single grid with grid lines"""
        if y_offset + self.max_grid_height * self.cell_size < self.scroll_y:
            return  # Skip if grid is above visible area
        if y_offset - self.scroll_y > self.window_height:
            return  # Skip if grid is below visible area
            
        height = len(grid)
        width = len(grid[0])
        
        for y in range(height):
            for x in range(width):
                color = COLORS.get(grid[y][x], (128, 128, 128))
                rect = pygame.Rect(
                    x_offset + x * self.cell_size,
                    y_offset - self.scroll_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, GRID_LINE)

    def draw_text(self, text, x, y):
        """Draw text on the surface if in visible area"""
        if y - self.scroll_y < -LABEL_HEIGHT or y - self.scroll_y > self.window_height:
            return
        
        font = pygame.font.SysFont('Arial', 24)
        text_surface = font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (x, y - self.scroll_y))

    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw training pairs
        y_offset = PADDING
        for i, pair in enumerate(self.puzzle['train']):
            self.draw_text(f'Training Pair {i+1}:', PADDING, y_offset)
            y_offset += LABEL_HEIGHT
            
            # Draw input grid
            self.draw_grid(pair['input'], PADDING, y_offset)
            
            # Draw output grid
            x_offset = self.max_grid_width * self.cell_size + (2 * PADDING)
            self.draw_grid(pair['output'], x_offset, y_offset)
            
            y_offset += (self.max_grid_height * self.cell_size) + PADDING
        
        # Draw test pair
        self.draw_text('Test:', PADDING, y_offset)
        y_offset += LABEL_HEIGHT
        
        # Draw test input grid
        self.draw_grid(self.puzzle['test'][0]['input'], PADDING, y_offset)
        
        # Draw test output grid
        x_offset = self.max_grid_width * self.cell_size + (2 * PADDING)
        self.draw_grid(self.puzzle['test'][0]['output'], x_offset, y_offset)
        
        self.draw_scrollbar()
        pygame.display.flip()

def main(filename, hide_test_output=False):
    # Load the puzzle
    with open(filename, 'r') as f:
        puzzle = json.load(f)
    
    # If hiding test output, create an empty grid of the same dimensions
    if hide_test_output and puzzle['test']:
        input_height = len(puzzle['test'][0]['input'])
        input_width = len(puzzle['test'][0]['input'][0]) if input_height > 0 else 0
        
        if 'output' in puzzle['test'][0]:
            output_height = len(puzzle['test'][0]['output'])
            output_width = len(puzzle['test'][0]['output'][0]) if output_height > 0 else 0
            # Create empty grid (filled with zeros)
            empty_grid = [[0 for _ in range(output_width)] for _ in range(output_height)]
            puzzle['test'][0]['output'] = empty_grid
    
    pygame.display.set_caption('ARC Puzzle Visualizer')
    
    window = ScrollableWindow(puzzle)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            window.handle_event(event)
        
        window.draw()
    
    pygame.quit()

if __name__ == '__main__':
    hide_test_output = False
    
    # Check for -n flag
    if "-n" in sys.argv:
        hide_test_output = True
        sys.argv.remove("-n")
    
    if len(sys.argv) != 2:
        print("Usage: python arcvis.py [-n] <puzzle_file.json>")
        print("  -n: Hide test output and display empty grid instead")
        sys.exit(1)
        
    main(sys.argv[1], hide_test_output)
