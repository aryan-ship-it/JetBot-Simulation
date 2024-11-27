import pygame
import sys
import heapq
import random
import pandas as pd
import os
import time


# Define colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # Thief color
YELLOW = (255, 215, 0)  # Coin color
BLUE = (0, 0, 255)  # Goal color

# Ensure you adjust `CELL_SIZE` to fit the new maze into the screen dimensions
CELL_SIZE = 20
WIDTH, HEIGHT = 30 * CELL_SIZE, 30 * CELL_SIZE

DATA_FILE = 'data_collector/maze_data.csv'

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Coin Collector")

maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# positions for AI, thieves, coins, and goal
ai_start = (1, 1)
coin_positions = [
    (9, 16), (18, 8), (25, 22), (5, 26), (7, 8), (16, 18), (22, 20), (5, 1), (18, 1), (26, 1)
]  # 10 coins
goal_position = (26, 28)


# Function to draw the maze
def draw_maze(thief_positions):
    for row in range(len(maze)):
        for col in range(len(maze[row])):
            color = BLACK if maze[row][col] == 1 else WHITE
            pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw coins
    for coin in coin_positions:
        coin_x = coin[1] * CELL_SIZE
        coin_y = coin[0] * CELL_SIZE
        pygame.draw.circle(screen, YELLOW, (coin_x + CELL_SIZE // 2, coin_y + CELL_SIZE // 2), CELL_SIZE // 4)

    # Draw thieves
    for thief in thief_positions:
        thief_x = thief[1] * CELL_SIZE
        thief_y = thief[0] * CELL_SIZE
        pygame.draw.rect(screen, RED, (thief_x, thief_y, CELL_SIZE, CELL_SIZE))

    # Draw goal
    goal_x = goal_position[1] * CELL_SIZE
    goal_y = goal_position[0] * CELL_SIZE
    pygame.draw.rect(screen, BLUE, (goal_x, goal_y, CELL_SIZE, CELL_SIZE))


# Heuristic for A* search (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# A* search algorithm with dynamic thief avoidance
def a_star_search(start, goal, thief_positions):
    frontier = []  # Min-Heap that stores the shortest next step to take
    heapq.heappush(frontier, (0, start))  # Initialize min heap with start
    came_from = {start: None}  # Map to keep track of where it has gone so far
    cost_so_far = {start: 0}  # Map to keep track of the cost to get to that specific position

    danger_cost = 50  # high penalty for being near thieves
    safe_distance = 5  # avoid thiefs that are within 5 steps

    while frontier:
        current = heapq.heappop(frontier)[1]  # Get the next move

        if current == goal:
            break

        # Explore neighbors (up, down, left, right)
        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            # Check if the neighbor is walkable
            if 0 <= neighbor[0] < len(maze) and 0 <= neighbor[1] < len(maze[0]) and maze[neighbor[0]][neighbor[1]] == 0:
                new_cost = cost_so_far[current] + 1  # Base cost to move to neighbor

                # Adjust cost based on proximity to thieves
                for thief in thief_positions:
                    distance_to_thief = heuristic(neighbor, thief)  # Calculate the distance to each thief
                    if distance_to_thief <= safe_distance:  # If the thief is in danger zone at that position
                        # Calculate a higher cost to avoid going to the thief  (Weight * distance to thief)
                        new_cost += danger_cost * (safe_distance - distance_to_thief + 1)

                # If the neighbor have not been visited or the new cost is lower than previous cost at that position
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    # Update the cost to go to that position
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(goal, neighbor)  # Calculate new priortity
                    heapq.heappush(frontier, (priority, neighbor))  # Push it to the min heap
                    came_from[neighbor] = current  # Update where it came from to get there

    # If goal is not in came_from, no valid path was found
    if goal not in came_from:
        return []

    # Reconstruct path
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def move_thieves(thief_positions, maze, last_moves):
    """Move each thief to a new position, avoiding backtracking unless at a dead end."""
    new_thief_positions = []
    updated_last_moves = []

    # Goes over each thief last move
    for i, thief in enumerate(thief_positions):
        last_move = last_moves[i]  # Get the last move of the current thief

        # Generate possible moves (up, down, left, right)
        possible_moves = [
            (thief[0] - 1, thief[1]),  # Up
            (thief[0] + 1, thief[1]),  # Down
            (thief[0], thief[1] - 1),  # Left
            (thief[0], thief[1] + 1)  # Right
        ]

        # Filter valid moves
        valid_moves = [
            move for move in possible_moves
            if 0 <= move[0] < len(maze) and 0 <= move[1] < len(maze[0])  # Within bounds
               and maze[move[0]][move[1]] != 1  # Not a wall
               and move not in thief_positions  # Avoid overlapping with other thieves
        ]

        # Exclude the last move from valid options if there are other valid moves
        if last_move in valid_moves and len(valid_moves) > 1:
            valid_moves.remove(last_move)

        # Decide on movement strategy (random, chasing AI, etc.)
        if valid_moves:
            # Random movement
            new_position = random.choice(valid_moves)
            new_thief_positions.append(new_position)
            updated_last_moves.append(thief)  # Update the last move to the current position
        else:
            # No valid moves, stay in place
            new_thief_positions.append(thief)
            updated_last_moves.append(last_move)  # Keep the last move as is

    return new_thief_positions, updated_last_moves

def load_or_initialize_data():
    """
    Loads the maze data from a CSV file or initializes a new DataFrame if the file doesn't exist.
    """
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        # Initialize a new DataFrame if the file doesn't exist
        return pd.DataFrame(columns=["Steps", "Completed_Maze", "Time_Taken"])

def save_data(dataframe):
    """
    Saves the DataFrame to a CSV file.
    """
    dataframe.to_csv(DATA_FILE, index=False)


def log_maze_data(steps, completed, time_taken, df):
    """
    Logs data about the maze into the DataFrame.
    """
    # Convert the completed boolean to a string
    completed_str = "Yes" if completed else "No"
    # Create a new row of data
    new_row = {
        "Steps": steps,
        "Completed_Maze": completed_str,
        "Time_Taken": round(time_taken, 2),
    }
    # Append the new row to the DataFrame
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


def main():
    maze_data_df = load_or_initialize_data()
    current_position = ai_start
    coin_collected = set()
    thief_positions = [(9, 15), (18, 6), (25, 20), (5, 25), (22, 14)]  # Example thief starting positions
    last_moves = thief_positions[:]  # Initialize last_moves to be the starting positions of the thieves

    step_counter = 0  # Initialize the step counter
    completed_maze = False
    clock = pygame.time.Clock()
    start_time = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Move thieves
        thief_positions, last_moves = move_thieves(thief_positions, maze, last_moves)

        if current_position in thief_positions:
            print(f"Collision with a thief at {current_position}! Game over.")
            break

        # Draw the maze, coins, and thieves
        screen.fill(WHITE)
        draw_maze(thief_positions)

        # Update AI logic
        remaining_coins = [coin for coin in coin_positions if coin not in coin_collected]

        if remaining_coins:
            closest_coin = min(remaining_coins, key=lambda coin: heuristic(current_position, coin))
            path_to_coin = a_star_search(current_position, closest_coin, thief_positions)

            if path_to_coin:
                next_step = path_to_coin[1]
                current_position = next_step
                if current_position in thief_positions:
                    print(f"Collision with a thief at {current_position}! Game over.")

                    break
                step_counter += 1

                if current_position == closest_coin:
                    coin_collected.add(closest_coin)
                    print(f"Collected coin at {closest_coin}")

                    # Replace coin with a path (0) in the maze
                    maze[closest_coin[0]][closest_coin[1]] = 0
                    # Update the coin_positions to path when the coin is collected
                    coin_positions.remove(closest_coin)
        else:
            path_to_goal = a_star_search(current_position, goal_position, thief_positions)

            if path_to_goal:
                next_step = path_to_goal[1]
                current_position = next_step
                if current_position in thief_positions:
                    print(f"Collision with a thief at {current_position}! Game over.")
                    break
                step_counter += 1

                if current_position == goal_position:
                    print(f"AI reached the finish in {step_counter} steps!")
                    completed_maze = True
                    break
            else:
                print("No reachable path to goal!")
                break

        # Draw the AI at its current position
        ai_x = current_position[1] * CELL_SIZE
        ai_y = current_position[0] * CELL_SIZE
        pygame.draw.rect(screen, GREEN, (ai_x, ai_y, CELL_SIZE, CELL_SIZE))

        # Update the screen
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(5)

    end_time = time.time()
    time_taken = end_time - start_time

    maze_data_df = log_maze_data (step_counter,completed_maze,time_taken,maze_data_df)
    save_data(maze_data_df)
    print(maze_data_df)


if __name__ == "__main__":
    main()
