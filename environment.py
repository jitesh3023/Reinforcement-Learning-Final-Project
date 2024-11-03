import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from a_star import a_star_search

class WarehouseEnvironment(gym.Env):
    def __init__(self, grid_size, num_materials, num_obstacles):
        super(WarehouseEnvironment, self).__init__()
        self.grid_size = grid_size
        self.num_materials = num_materials
        self.num_obstacles = num_obstacles
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([
            grid_size, grid_size,  # Robot position
            grid_size, grid_size,  # Goal position
            grid_size, grid_size,  # Material positions
        ])
        self.robot_position = (0, 0)
        self.goal_position = (4, 4)
        self.material_positions = self.generate_random_positions(num_materials)
        self.obstacle_positions = self.generate_random_positions(num_obstacles)

    def generate_random_positions(self, num_positions):
        positions = set()
        while len(positions) < num_positions:
            position = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if position != self.robot_position and position != self.goal_position:
                positions.add(position)
        return list(positions)

    def reset(self):
        self.robot_position = (0, 0)
        return self.get_state()

    def get_state(self):
        return np.array([self.robot_position[0], self.robot_position[1], self.goal_position[0],
                         self.goal_position[1], *sum(self.material_positions, ())]).flatten()

    def is_valid_move(self, position):
        return 0 <= position[0] < self.grid_size and 0 <= position[1] < self.grid_size and position not in self.obstacle_positions

    def take_action(self, action):
        new_position = self.robot_position
        if action == 0:  # Go up
            new_position = (self.robot_position[0] - 1, self.robot_position[1])
        elif action == 1:  # Go down
            new_position = (self.robot_position[0] + 1, self.robot_position[1])
        elif action == 2:  # Go left
            new_position = (self.robot_position[0], self.robot_position[1] - 1)
        elif action == 3:  # Go right
            new_position = (self.robot_position[0], self.robot_position[1] + 1)
        if self.is_valid_move(new_position):
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
        if self.collect_material():
            reward = 3
        else:
            reward = 0

        done = len(self.material_positions) == 0 and self.robot_position == self.goal_position
        return self.get_state(), reward, done, {}

    def render(self, mode, current_position=None):
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
                obstacle_rect = Rectangle((obstacle_position[1], obstacle_position[0]), 1, 1,
                                        linewidth=1, edgecolor='black', facecolor='black')
                ax.add_patch(obstacle_rect)

            for material_position in self.material_positions:
                material_rect = Rectangle((material_position[1], material_position[0]), 1, 1,
                                        linewidth=1, edgecolor='green', facecolor='green')
                ax.add_patch(material_rect)

            goal_rect = Rectangle((self.goal_position[1], self.goal_position[0]), 1, 1,
                                linewidth=1, edgecolor='red', facecolor='red')
            ax.add_patch(goal_rect)

            if current_position:
                robot_rect = Rectangle((current_position[1], current_position[0]), 1, 1,
                                    linewidth=1, edgecolor='blue', facecolor='blue')
                ax.add_patch(robot_rect)
            else:
                robot_rect = Rectangle((self.robot_position[1], self.robot_position[0]), 1, 1,
                                    linewidth=1, edgecolor='blue', facecolor='blue')
                ax.add_patch(robot_rect)

            ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
            ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
            ax.grid(which='both', color='black', linestyle='-', linewidth=0.7)

            plt.show()

    def collect_material(self):
        if self.robot_position in self.material_positions:
            self.material_positions.remove(self.robot_position)
            return True
        return False

    def find_optimal_path(self):
        return a_star_search(self)