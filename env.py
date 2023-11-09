import gym
from gym import spaces
import numpy as np

class WarehouseEnvironment(gym.Env):
    def __init__(self):
        super(WarehouseEnvironment, self).__init__()
        # Defining the warehouse parametesr
        self.num_items = num_items
        self.num_robots = num_robots 
        self.max_steps = 100

        self.action_space = spaces.Discrete(5)
        #self.observation_space
        self.warehouse_observation_space = spaces.Box(low=-2.0, high=2.0, shape=(10,10), dtype=np.float32)
        self.warehouse_state = np.zeros(self.num_items+self.num_robots)
        self.reset()

    def reset(self):
        self.warehouse_state = np.zeros(self.num_items+self.num_robots)
        self.current_step = 0
        return self.warehouse_state
    
    def step(self, action):
        reward = 