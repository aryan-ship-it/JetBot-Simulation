import pygame
import sys
import heapq

# Initialize Pygame
pygame.init()

# Define some colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)  # For coins
BLUE = (0, 0, 255)      # For walls
GREEN = (0, 255, 0)     # For the player/AI
PURPLE = (160, 32, 240) # For finish points

# Set the size of the grid cells and the window size
CELL_SIZE = 40
GRID_WIDTH = 15
GRID_HEIGHT = 12
WINDOW_WIDTH = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT

# Create the screen
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Maze with AI Pathfinding")

# Define the maze layout (0 = free space, 1 = wall, 2 = coin, 3 = obstacle, 4 = finish point)
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 4, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 4, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 2, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 4, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Player and AI starting position (row, col)
ai_start = (1, 1)

# Define finish points and coins
finish_points = [(1, 13), (7, 13), (10, 13)]  # Multiple finish points (row, col)
coin_positions = [(1, 4), (7, 7), (5, 4)]  # Coins' positions

# A* Search implementation with priority to coin collection
def a_star_search(start, goal):
    """A* algorithm to find the path from start to goal"""
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        # If the current node is the goal, reconstruct path
        if current == goal:
            return reconstruct_path(came_from, current)
        
        # Get neighbors
        neighbors = get_neighbors(current)
        
        for neighbor in neighbors:
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []  # Return empty path if no solution

def get_neighbors(pos):
    """Returns the valid neighbors of the current position"""
    row, col = pos
    neighbors = []
    
    for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + d_row, col + d_col
        
        if 0 <= new_row < GRID_HEIGHT and 0 <= new_col < GRID_WIDTH and maze[new_row][new_col] != 1:
            neighbors.append((new_row, new_col))
    
    return neighbors

def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, current):
    """Reconstruct the path from start to goal"""
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

def find_path_with_coins(start):
    """Find the most optimal path that collects the maximum coins"""
    path = []
    remaining_coins = set(coin_positions)
    current_position = start
    
    while remaining_coins:
        # Find the nearest coin
        nearest_coin = min(remaining_coins, key=lambda coin: heuristic(current_position, coin))
        
        # Find path to the nearest coin
        path_to_coin = a_star_search(current_position, nearest_coin)
        path += path_to_coin[1:]  # Skip the first position (already there)
        
        # Move to that coin and collect it
        current_position = nearest_coin
        remaining_coins.remove(nearest_coin)
    
    # After collecting all coins, find the nearest finish point
    nearest_finish = min(finish_points, key=lambda finish: heuristic(current_position, finish))
    path_to_finish = a_star_search(current_position, nearest_finish)
    path += path_to_finish[1:]
    
    return path

def draw_maze():
    """Function to draw the maze based on the grid layout"""
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            cell = maze[row][col]
            x = col * CELL_SIZE
            y = row * CELL_SIZE

            if cell == 1:  # Wall
                pygame.draw.rect(screen, BLUE, (x, y, CELL_SIZE, CELL_SIZE))
            elif cell == 2:  # Coin
                pygame.draw.circle(screen, YELLOW, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 4)
            elif cell == 3:  # Obstacle
                pygame.draw.rect(screen, RED, (x, y, CELL_SIZE, CELL_SIZE))
            elif cell == 4:  # Finish points
                pygame.draw.rect(screen, PURPLE, (x, y, CELL_SIZE, CELL_SIZE))
            else:  # Empty space
                pygame.draw.rect(screen, WHITE, (x, y, CELL_SIZE, CELL_SIZE))

def draw_ai(ai_path, step):
    """Draw the AI on the screen at the current step in its path"""
    if step < len(ai_path):
        ai_x = ai_path[step][1] * CELL_SIZE
        ai_y = ai_path[step][0] * CELL_SIZE
        pygame.draw.rect(screen, GREEN, (ai_x, ai_y, CELL_SIZE, CELL_SIZE))

def main():
    # Find the optimal path for the AI to collect coins and reach the finish
    ai_path = find_path_with_coins(ai_start)
    
    clock = pygame.time.Clock()
    step = 0
    
    # Main loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Draw the maze
        screen.fill(WHITE)
        draw_maze()
        
        # Draw the AI
        draw_ai(ai_path, step)
        step += 1
        if step >= len(ai_path):
            step = len(ai_path) - 1  # Stop at the end
        
        # Update the screen
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(5)

if __name__ == "__main__":
    main()
