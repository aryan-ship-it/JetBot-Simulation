import pygame
import sys
import heapq
import random

# Define colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # Thief color
YELLOW = (255, 215, 0)  # Coin color
BLUE = (0, 0, 255)  # Goal color

# Define maze dimensions
CELL_SIZE = 40
WIDTH, HEIGHT = 600, 600

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Coin Collector")

# Maze data (1 represents wall, 0 represents walkable path)
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Define start position, thieves, coins, and goal
ai_start = (1, 1)
thief_positions = [(9,3), (2,11), (11,7)]  # 3 thieves
coin_positions = [(1,9), (5,11), (9,5), (13,4), (9,11)]  # 5 coins
goal_position = (13, 13)  # Goal at (13, 13)



# Function to draw the maze
def draw_maze():
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

# A* search algorithm
def a_star_search(start, goal, thief_positions):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while frontier:
        current = heapq.heappop(frontier)[1]

        if current == goal:
            break

        # Explore neighbors (up, down, left, right)
        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            # Check if the neighbor is walkable
            if 0 <= neighbor[0] < len(maze) and 0 <= neighbor[1] < len(maze[0]) and maze[neighbor[0]][neighbor[1]] == 0 and neighbor not in thief_positions:
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(goal, neighbor)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

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

# Main function
def main():
    current_position = ai_start
    coin_collected = set()
    
    clock = pygame.time.Clock()
    heading_to_exit = False  # Flag to track if the AI is heading to an exit
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Draw the maze and objects
        screen.fill(WHITE)
        draw_maze()

        # If there are remaining coins, find and move toward the nearest one
        if not heading_to_exit:
            remaining_coins = [coin for coin in coin_positions if coin not in coin_collected]
            
            if remaining_coins:
                nearest_coin = min(remaining_coins, key=lambda coin: heuristic(current_position, coin))
                path_to_coin = a_star_search(current_position, nearest_coin, thief_positions)

                if path_to_coin:  # If a valid path to the coin is found
                    next_step = path_to_coin[1]  # Move to the next step
                    current_position = next_step

                    # If the AI reaches the coin, collect it
                    if current_position == nearest_coin:
                        coin_collected.add(nearest_coin)
                        print(f"Collected coin at {nearest_coin}")
            else:
                # No remaining coins, switch to heading towards an exit
                heading_to_exit = True
                print("All coins collected! Heading towards the nearest exit...")

        # If heading to an exit, find and move toward the nearest one
        if heading_to_exit:
            path_to_exit = a_star_search(current_position, goal_position, thief_positions)
                
            if path_to_exit:
                next_step = path_to_exit[1]
                current_position = next_step

                # If the AI reaches the exit, stop the game
                if current_position == goal_position:
                    print("AI reached the finish!")
                    break
            else:
                print("No reachable exits found!")
                break

        # Draw the AI at its current position
        ai_x = current_position[1] * CELL_SIZE
        ai_y = current_position[0] * CELL_SIZE
        pygame.draw.rect(screen, GREEN, (ai_x, ai_y, CELL_SIZE, CELL_SIZE))

        # Update the screen
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(5)

if __name__ == "__main__":
    main()
