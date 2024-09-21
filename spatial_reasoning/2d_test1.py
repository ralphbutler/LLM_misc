import sys, os, time, textwrap

import query_all

import pygame
from pydantic import BaseModel, Field

OLLAMA_MODEL    = "llama3.1"
OPENAI_MODEL    = "gpt-4o-2024-08-06"
# GOOGLE_MODEL    = "gemini-1.5-pro-exp-0827"
GOOGLE_MODEL    = "gemini-1.5-flash-exp-0827"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"

SPATIAL_MODEL = OPENAI_MODEL

# Initialize Pygame
pygame.init()

# Set the window size
window_size = (400, 320)
screen = pygame.display.set_mode(window_size)

# Set the title
pygame.display.set_caption("Grid of Squares")

# Define colors
green = (0, 255, 0)
red = (255, 0, 0)
black = (0, 0, 0) # For grid lines
blue = (0, 0, 255)
orange = (255, 165, 0)

# Square size
square_size = 80

# Function to draw a circle at a specific grid position
def draw_circle_at_position(screen, row, col):
    x = col * square_size + square_size // 2
    y = row * square_size + square_size // 2
    circle_radius = square_size // 2 - 5  # Slightly smaller than the square
    pygame.draw.circle(screen, blue, (x, y), circle_radius)

# Function to draw the grid
def draw_grid(screen, red_locs, circle_loc, destination_loc):
    # Fill the background
    screen.fill((255, 255, 255))

    # Draw the grid
    for row in range(4):
        for col in range(5):
            # Calculate square position
            x = col * square_size
            y = row * square_size

            # Set color based on position
            if (row, col) == destination_loc:
                color = orange
            elif (row, col) in red_locs:
                color = red
            else:
                color = green

            # Draw the square
            pygame.draw.rect(screen, color, (x, y, square_size, square_size))

            # Draw grid lines (optional)  1 is line-width
            pygame.draw.rect(screen, black, (x, y, square_size, square_size), 1)

    # Draw the blue circle
    draw_circle_at_position(screen, circle_loc[0], circle_loc[1])

    # Update the display
    pygame.display.flip()

    return screen

class Move(BaseModel):
    move: str = Field(description="the next move to make in the game (up, down, left, right)")

prompt = textwrap.dedent("""
We want to move the blue circle from its current square to the orange square.
<rules>
    1. take the shortest path
    2. do NOT move onto a red square
</rules>
<possible_moves>
    - up
    - down
    - left
    - right
</possible_moves>

It is OK to move onto the orange square; that is the goal.

Now, choose the next valid move from the set of possible moves.
""")

if "gpt-" in SPATIAL_MODEL:
    extra_prompt = textwrap.dedent("""
        Produce the result in JSON that matches this schema:
            {
                "move": "the move",
            }
    """)
    prompt += extra_prompt

image_url = os.path.join(os.getcwd(), "current_board.jpeg")

image_messages = [
    {
        "role": "system",
        "content": "You are an expert at using an image to plan moves in a 2D space.",

        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    },
]


# main game loop
def main_game_loop():
    running = True
    red_locs = [(2, 0), (2, 2), (2, 3)]  # Initial red squares
    circle_loc = (0, 0)  # Initial blue circle location
    destination_loc = (3, 3)  # Initial orange square location
    invalid_move = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        updated_screen = draw_grid(screen, red_locs, circle_loc, destination_loc)
        pygame.image.save(updated_screen, image_url)

        if not invalid_move:
            print("calling LLM...")
            response = query_all.generate(SPATIAL_MODEL, image_messages, json_object=True)
            print("DBGRESPONSE", response)
            move_data = Move.parse_raw(response)
            move = move_data.move
        else:
            continue

        # Calculate new position based on the move
        new_circle_loc = list(circle_loc)
        if move == "up":
            new_circle_loc[0] -= 1
        elif move == "down":
            new_circle_loc[0] += 1
        elif move == "left":
            new_circle_loc[1] -= 1
        elif move == "right":
            new_circle_loc[1] += 1
        else:
            print(f"Error: Invalid move '{move}'. Expected 'up', 'down', 'left', or 'right'.")
            invalid_move = move
            # running = False
            # break

        # Check if the new position is out of bounds
        if (
            new_circle_loc[0] < 0 or new_circle_loc[0] > 3 or
            new_circle_loc[1] < 0 or new_circle_loc[1] > 4
        ):
            print(f"Error: Invalid move to position {tuple(new_circle_loc)}.")
            print("The move is out of bounds.")
            invalid_move = "out_of_bounds"
            # running = False
            # break

        # Check if the new position is on a red square
        if tuple(new_circle_loc) in red_locs:
            print(f"Error: Invalid move to position {tuple(new_circle_loc)}.")
            print("The move is onto a red square.")
            invalid_move = "onto_red_square"
            # running = False
            # break

        # Update the circle's location
        circle_loc = tuple(new_circle_loc)

        # Check if the circle has reached the destination
        if circle_loc == destination_loc:
            print("Congratulations! The blue circle has reached the orange square.")
            # running = False
            # break

        time.sleep(1)  # to watch on-screen

    # os.remove(image_url)
    pygame.quit()

if __name__ == "__main__":
    main_game_loop()
