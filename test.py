import gym
import rware
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

agent_count = 1
env = gym.make(f"rware-tiny-{agent_count}ag-v1")


obs1 = env.observation_space.sample()  # the observation space can be sampled
print(obs1[0].shape ) 
print(obs1[0] ) 

obs2 = env.reset()  # a tuple of observations
print(obs2[0].shape ) 
print(obs2[0]) 

print(env.grid)

env.render()