import gym
import rware
env = gym.make("rware-tiny-2ag-v1")

# Enviroment variables can be accessed 


obs = env.reset()  # a tuple of observations

actions = env.action_space.sample()  # the action space can be sampled
print(actions)  # (1, 0)
n_obs, reward, done, info = env.step(actions)

print(done)    # [False, False]
print(reward)  # [0.0, 0.0]

env.render()

env.close()