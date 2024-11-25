import numpy as np
import random
from queue import PriorityQueue

# Maze definition (0: free, 1: wall)
# maze = np.array([
#     [0, 1, 0, 0, 0],
#     [0, 1, 0, 1, 0],
#     [0, 0, 0, 1, 0],
#     [0, 1, 0, 0, 0],
#     [0, 0, 0, 1, 0]
# ])
# [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (3, 3), (3, 4), (4, 4)]

# maze = np.array([
#     [0, 1, 0, 0, 0],
#     [0, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0]
# ])
# [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 4)]

maze = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
])
# Test Path Found by RL Agent: [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (3, 2), (3, 3), (3, 4), (4, 4)]
# episodes = 5, reward 3


maze = np.array([
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 0]
])


start = (0, 0)  # Start position
# goal = (4, 4)   # Goal position
goal = (6,6)

# Hyperparameters for RL
alpha = 0.1    # Learning rate
gamma = 0.9    # Discount factor
epsilon = 0.4  # Exploration rate
episodes = 1

# Initialize Q-table
rows, cols = maze.shape
# q_table = np.zeros((rows, cols, 4))  # Four actions per state

# Heuristic for A* (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* search
def a_star_search(maze, start, goal):
    rows, cols = maze.shape
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_set.empty():
        _, current = open_set.get()
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (x + dx, y + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor] != 1:
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.put((f_score[neighbor], neighbor))
    return []

# Use A* to compute the initial path
a_star_path = a_star_search(maze, start, goal)
print("A* Path:", a_star_path)

import numpy as np

def initialize_q_table(maze):
    rows, cols = maze.shape
    actions = 4  # Up, Down, Left, Right
    q_table = np.zeros((rows, cols, actions))  # Initialize all Q-values to 0

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    # Loop through each state in the maze
    for x in range(rows):
        for y in range(cols):
            if maze[x, y] == 1:  # Wall
                q_table[x, y, :] = -np.inf  # All actions invalid at walls
            else:
                for action, (dx, dy) in enumerate(moves):
                    nx, ny = x + dx, y + dy
                    # Check if the resulting move is out of bounds or hits a wall
                    if not (0 <= nx < rows and 0 <= ny < cols) or maze[nx, ny] == 1:
                        q_table[x, y, action] = -np.inf  # Set invalid actions to -infinity

    return q_table

q_table = initialize_q_table(maze)
print("qtable initialized\n", q_table)

# RL functions
def choose_action(state):
    if random.random() < epsilon:  # Exploration
        return random.choice(range(4))
    else:  # Exploitation
        x, y = state
        # print(q_table[x, y])
        # print(x,y)
        # print(np.argmax(q_table[x, y]))
        if y != 0:
            print(x,y)

        return np.argmax(q_table[x, y])

def take_action(state, action):
    x, y = state
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    dx, dy = moves[action]
    next_state = (x + dx, y + dy)
    print("state",state,"move",moves[action],"next state",next_state)

    if 0 <= next_state[0] < rows and 0 <= next_state[1] < cols and maze[next_state] != 1:
        return next_state, -1  # Valid move, step penalty
    return state, -10  # Invalid move, heavy penalty

# Training loop
for episode in range(episodes):
    print("episode", episode)
    state = start
    total_reward = 0
    visited_states = set(a_star_path)  # Use A* path as guidance

    while state != goal:
        action = choose_action(state)
        # print("action", action)
        next_state, reward = take_action(state, action)

        # Boost reward for following A* path
        if next_state in visited_states:
            # reward += 5  # Bonus for staying on A* path TODO can be lower
            reward +=3

        # Q-learning update
        x, y = state
        nx, ny = next_state
        if (q_table[x, y, action] != -np.inf):
            q_table[x, y, action] += alpha * (reward + gamma * np.max(q_table[nx, ny]) - q_table[x, y, action])

        state = next_state
        total_reward += reward

    # Decay exploration rate over time
    epsilon = max(0.1, epsilon * 0.995)



print("Training Complete!")

# Testing the RL agent
def test_agent(maze, q_table, start, goal):
    print("qtable\n",q_table)
    state = start
    path = [state]
    visited = set()  # Track visited states to prevent loops

    while state != goal:
        print("state", state)
        print("path", path)
        x, y = state
        
        while True:
            action = np.argmax(q_table[x, y])
            print("qtable[x,y]",q_table[x, y])
            print("action",action)
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dx, dy = moves[action]
            print("dx dy",dx,dy)
            next_state = (x + dx, y + dy)
            print("next state",next_state)
            if next_state not in visited:
                print("next state not visited")
                break
            else:
                print("next state visited")
                # Temporarily set the action's Q-value to -inf to avoid selecting it again
                temp_max = q_table[x, y, action]
                temp_x = x
                temp_y = y
                temp_action = action
                q_table[x, y, action] = -np.inf

        
        # Validate the next state
        if (
            0 <= next_state[0] < maze.shape[0] and
            0 <= next_state[1] < maze.shape[1] and
            maze[next_state] != 1 and
            next_state not in visited
        ):
            state = next_state
            path.append(state)
            visited.add(state)  # Add to visited to prevent revisiting
        else:
            # If the agent is stuck (invalid move or loop), break the loop
            print(f"Agent is stuck at state {state} with action {action}")
            break
    return path



# Test the trained agent
test_path = test_agent(maze, q_table, start, goal)
print("Test Path Found by RL Agent:", test_path)
print("A* Path:", a_star_path)
