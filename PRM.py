import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.neighbors import KDTree

class WarehouseEnvironment(gym.Env):
    def __init__(self, grid_size, num_materials, num_obstacles, prm_k=5):
        super(WarehouseEnvironment, self).__init__()
        self.grid_size = grid_size
        self.num_materials = num_materials
        self.num_obstacles = num_obstacles
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([
            grid_size, grid_size,
            grid_size, grid_size,
            grid_size, grid_size,
        ])
        self.robot_position = (0, 0)
        self.goal_position = (1, 0)
        self.material_positions = self.generate_random_positions(num_materials)
        self.obstacle_positions = self.generate_random_positions(num_obstacles)
        self.prm_k = prm_k
        self.prm_graph = self.build_prm()

    def generate_random_positions(self, num_positions):
        positions = set()
        while len(positions) < num_positions:
            position = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if position != self.robot_position and position != self.goal_position:
                positions.add(position)
        return list(positions)

    def build_prm(self):
        prm_graph = {}
        all_positions = [self.robot_position, self.goal_position] + self.material_positions + self.obstacle_positions
        kdtree = KDTree(all_positions)

        for position in all_positions:
            _, indices = kdtree.query(np.array(position).reshape(1, -1), k=self.prm_k + 1)
            prm_graph[position] = [all_positions[i] for i in indices[1:]]

        return prm_graph


    def prm_plan(self, start, goal):
        # Implement a simple path finding using the PRM graph
        queue = [(start, [])]
        visited = set()

        while queue:
            current, path = queue.pop(0)
            if current == goal:
                return path + [goal]

            visited.add(current)
            neighbors = self.prm_graph[current]

            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, path + [current]))

        return []

    def reset(self):
        self.robot_position = (0, 0)
        return self.get_state()

    def get_state(self):
        return np.array([self.robot_position[0], self.robot_position[1], self.goal_position[0],
                         self.goal_position[1], *sum(self.material_positions, ())]).flatten()

    def is_valid_move(self, position):
        return 0 <= position[0] < self.grid_size and 0 <= position[1] < self.grid_size and position not in self.obstacle_positions

    def take_action(self, action):
        prm_path = self.prm_plan(self.robot_position, self.goal_position)

        if prm_path:
            new_position = prm_path[1]
            self.robot_position = new_position

    def is_goal_reached(self):
        return self.robot_position == self.goal_position

    def collect_material(self):
        if self.robot_position in self.material_positions:
            self.material_positions.remove(self.robot_position)
            return True
        return False

    def step(self, action):
        self.previous_position = self.robot_position
        self.take_action(action)

        if self.robot_position == self.goal_position:
            if len(self.material_positions) == 0:
                reward = 5
            else:
                reward = 0
                self.robot_position = self.previous_position
        elif self.collect_material():
            reward = 3
        else:
            reward = 0

        done = self.is_goal_reached() and len(self.material_positions) == 0
        return self.get_state(), reward, done, {}

    def render(self, mode):
        if mode == 'human':
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    position = (i, j)
                    if position == self.robot_position:
                        print('R', end=' ')
                    elif position == self.goal_position:
                        print('G', end=' ')
                    elif position in self.material_positions:
                        print('M', end=' ')
                    elif position in self.obstacle_positions:
                        print('O', end=' ')
                    else:
                        print('-', end=' ')
                print()
        elif mode == 'matplotlib':
            fig, ax = plt.subplots()
            ax.set_xlim([0, self.grid_size])
            ax.set_ylim([0, self.grid_size])

            ax.invert_yaxis()

            for obstacle_position in self.obstacle_positions:
                obstacle_rect = Rectangle((obstacle_position[1], obstacle_position[0]), 1, 1, linewidth=1,
                                          edgecolor='black', facecolor='black')
                ax.add_patch(obstacle_rect)

            for material_position in self.material_positions:
                material_rect = Rectangle((material_position[1], material_position[0]), 1, 1, linewidth=1,
                                          edgecolor='green', facecolor='green')
                ax.add_patch(material_rect)

            goal_rect = Rectangle((self.goal_position[1], self.goal_position[0]), 1, 1, linewidth=1,
                                  edgecolor='red', facecolor='red')
            ax.add_patch(goal_rect)

            robot_rect = Rectangle((self.robot_position[1], self.robot_position[0]), 1, 1, linewidth=1,
                                   edgecolor='blue', facecolor='blue')
            ax.add_patch(robot_rect)

            ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
            ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
            ax.grid(which='both', color='black', linestyle='-', linewidth=0.7)

            plt.show()

# Example usage
env = WarehouseEnvironment(grid_size=5, num_materials=3, num_obstacles=5)
env.render(mode='matplotlib')
for step in range(10):
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    env.render(mode='matplotlib')

    print(f"Step: {step + 1}")
    print(f"Action: {action}")
    print(f"State: {state}")
    print(f"Reward: {reward}")
    print(f"Materials left: {len(env.material_positions)}")

    if done:
        print("Goal reached!")
        break