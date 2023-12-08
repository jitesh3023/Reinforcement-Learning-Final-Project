import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx

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
        self.robot_position = (0,0)
        self.goal_position = (1,0)
        self.material_positions = self.generate_random_positions(num_materials)
        self.obstacle_positions = self.generate_random_positions(num_obstacles)
        self.prm_graph = None
        self.prm_nodes = 50
        self.prm_k_neighbors = 5
        self.build_prm()

    def generate_random_positions(self, num_positions):
        positions = set()
        while len(positions) < num_positions:
            position = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if position!=self.robot_position and position!=self.goal_position:
                positions.add(position)
        return list(positions)
    
    def reset(self):
        self.robot_position = (0, 0)
        self.robot_trajectory = []  # Reset robot trajectory
        self.build_prm()
        return self.get_state()
    
    def get_state(self):
        return np.array([self.robot_position[0], self.robot_position[1], self.goal_position[0], self.goal_position[1],*sum(self.material_positions, ())]).flatten()

    def is_valid_move(self, position):
        return 0 <= position[0] < self.grid_size and 0 <= position[1] < self.grid_size and position not in self.obstacle_positions
    
    def take_action(self, action):
        new_position = self.robot_position
        if action == 0: #Go up
            new_position=(self.robot_position[0]-1, self.robot_position[1])
        elif action == 1: #Go down
            new_position=(self.robot_position[0]+1, self.robot_position[1])
        elif action == 2: #Go left
            new_position=(self.robot_position[0], self.robot_position[1]-1)
        elif action == 3: #Go right
            new_position=(self.robot_position[0], self.robot_position[1]+1)
        if self.is_valid_move(new_position):
            self.robot_position = new_position

    def is_goal_reached(self):
        return self.robot_position == self.goal_position
    
    def collect_material(self):
        if self.robot_position in self.material_positions:
            self.material_positions.remove(self.robot_position)
            return True
        return False
    
    def build_prm(self):
        self.prm_graph = nx.random_geometric_graph(self.prm_nodes, radius=0.2)
        start_node = (self.robot_position[0], self.robot_position[1])
        goal_node = (self.goal_position[0], self.goal_position[1])
        self.prm_graph.add_node('start', pos=start_node)
        self.prm_graph.add_node('goal', pos=goal_node)

        for node in self.prm_graph.nodes():
            if node != 'start' and node != 'goal':
                distance_start = np.linalg.norm(np.array(start_node) - np.array(self.prm_graph.nodes[node]['pos']))
                distance_goal = np.linalg.norm(np.array(goal_node) - np.array(self.prm_graph.nodes[node]['pos']))
                if distance_start < 2.0:
                    self.prm_graph.add_edge('start', node)
                if distance_goal < 2.0:
                    self.prm_graph.add_edge('goal', node)

        for edge in self.prm_graph.edges():
            distance = np.linalg.norm(np.array(self.prm_graph.nodes[edge[0]]['pos']) - np.array(self.prm_graph.nodes[edge[1]]['pos']))
            if distance < 2.0:
                self.prm_graph.add_edge(edge[0], edge[1])

    def prm_path(self):
        try:
            prm_path = nx.shortest_path(self.prm_graph, 'start', 'goal')
            return prm_path
        except nx.NetworkXNoPath:
            return None

    def step(self, action):
        self.previous_position = self.robot_position
        self.take_action(action)
        self.robot_trajectory.append(self.robot_position)  # Store current robot position

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

        prm_path = self.prm_path()
        if prm_path:
            if len(prm_path) > 1:
                next_position = prm_path[1]
                self.robot_position = (next_position[0], next_position[1])

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

            print("Robot Trajectory:", self.robot_trajectory)
        elif mode == 'matplotlib':
            fig, ax = plt.subplots()
            ax.set_xlim([0, self.grid_size])
            ax.set_ylim([0, self.grid_size])

            for obstacle_position in self.obstacle_positions:
                obstacle_rect = Rectangle((obstacle_position[1], obstacle_position[0]), 1, 1, linewidth=1, edgecolor='black',
                                        facecolor='black')
                ax.add_patch(obstacle_rect)

            for material_position in self.material_positions:
                material_rect = Rectangle((material_position[1], material_position[0]), 1, 1, linewidth=1, edgecolor='green',
                                        facecolor='green')
                ax.add_patch(material_rect)

            goal_rect = Rectangle((self.goal_position[1], self.goal_position[0]), 1, 1, linewidth=1, edgecolor='red',
                                facecolor='red')
            ax.add_patch(goal_rect)

            for step, position in enumerate(self.robot_trajectory):
                robot_rect = Rectangle((position[1], position[0]), 1, 1, linewidth=1, edgecolor='blue', facecolor='blue')
                ax.add_patch(robot_rect)
                plt.pause(0.5)  # Pause for a moment to visualize each step

            ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
            ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
            ax.grid(which='both', color='black', linestyle='-', linewidth=0.7)

            plt.show()


    def render_prm(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim([0, self.grid_size])
        ax.set_ylim([0, self.grid_size])

        for edge in self.prm_graph.edges():
            pos1 = np.array(self.prm_graph.nodes[edge[0]]['pos'])
            pos2 = np.array(self.prm_graph.nodes[edge[1]]['pos'])
            plt.plot([pos1[1], pos2[1]], [pos1[0], pos2[0]], 'k-', linewidth=0.7)

        for node in self.prm_graph.nodes():
            pos = np.array(self.prm_graph.nodes[node]['pos'])
            plt.plot(pos[1], pos[0], 'bo', markersize=5)

        prm_path = self.prm_path()
        if prm_path:
            prm_path_positions = [np.array(self.prm_graph.nodes[node]['pos']) for node in prm_path]
            prm_path_positions = np.array(prm_path_positions)
            plt.plot(prm_path_positions[:, 1], prm_path_positions[:, 0], 'r-', linewidth=2, label='PRM Path')

        plt.legend()
        plt.show()

# Example usage:
env = WarehouseEnvironment(grid_size=5, num_materials=3, num_obstacles=3)
env.render(mode='matplotlib')
