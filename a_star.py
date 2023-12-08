import heapq
import numpy as np

class Node:
    def __init__(self, position, g_cost, h_cost, material_positions, parent=None):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.material_positions = material_positions
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost

def heuristic(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

def get_neighbors(position, grid_size):
    neighbors = []
    for action in range(4):
        new_position = position
        if action == 0:  # Go up
            new_position = (position[0] - 1, position[1])
        elif action == 1:  # Go down
            new_position = (position[0] + 1, position[1])
        elif action == 2:  # Go left
            new_position = (position[0], position[1] - 1)
        elif action == 3:  # Go right
            new_position = (position[0], position[1] + 1)

        if 0 <= new_position[0] < grid_size and 0 <= new_position[1] < grid_size:
            neighbors.append(new_position)

    return neighbors

def a_star_search(env):
    grid_size = env.grid_size
    start_node = Node(env.robot_position, 0, heuristic(env.robot_position, env.goal_position), env.material_positions)
    open_set = [start_node]
    closed_set = set()

    while open_set:
        current_node = heapq.heappop(open_set)
        if current_node.position == env.goal_position and not current_node.material_positions:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add((current_node.position, tuple(current_node.material_positions)))

        for neighbor_position in get_neighbors(current_node.position, grid_size):
            if (neighbor_position, tuple(current_node.material_positions)) not in closed_set \
                    and env.is_valid_move(neighbor_position):
                new_material_positions = current_node.material_positions.copy()
                if neighbor_position in new_material_positions:
                    new_material_positions.remove(neighbor_position)
                g_cost = current_node.g_cost + 1
                h_cost = heuristic(neighbor_position, env.goal_position)
                new_node = Node(neighbor_position, g_cost, h_cost, new_material_positions, current_node)
                heapq.heappush(open_set, new_node)

    return None
