import numpy as np
import env
from collections import defaultdict
import matplotlib.pyplot as plt
import random

def argmax(arr) -> int:
    """Argmax that breaks ties randomly

    Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

    Note: np.argmax returns the first index that matches the maximum, so we define this method to use in EpsilonGreedy and UCB agents.
    Args:
        arr: sequence of values
    """
    max_value = np.max(arr)
    max_indices = np.where(arr == max_value)[0]
    max_ = random.choice(max_indices)
    
    return max_

def monte_carlo(env, num_episodes, gamma=0.9):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for episode in range(num_episodes):
        state = env.reset()  # setting the robot/agent state to start
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        # Generating the episode with a policy
        done = False
        
        while not done:
            if np.random.rand() < 0.1:
                action = np.random.choice(env.action_space.n)
            else:
                action = argmax(Q[state])
            
            print(action)
            next_state, reward, done, _ = env.step(action)
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state
            print(f"Episode: {episode}, State: {state}, Action: {action}, Done: {done}")

        returns = []
        lengths = []
        G = 0
        visited_states = set()

        for t in range(len(episode_states) - 1, -1, -1):
            state_t = tuple(episode_states[t])  # Convert to tuple
            action_t = episode_actions[t]
            reward_t = episode_rewards[t]
            G = gamma * G + reward_t

            # Unless S_t appears in visited_states:
            if state_t not in visited_states:
                visited_states.add(state_t)
                N[state_t][action_t] += 1
                Q[state_t][action_t] += (1 / N[state_t][action_t]) * (G - Q[state_t][action_t])

    return Q, returns


def q_learning(
    env,
    num_episodes: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    def get_policy(state):
        if np.random.rand() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = argmax(Q[state])

        return action
    
    done = False
    returns = []
    lengths = []
    state = env.reset()

    for episode in range(num_episodes):
        state = env.reset()
        t_episode = 0
        done = False
        while not done:
            t_episode += 1
            action = get_policy(state)
            next_state, reward, done, _ = env.step(action)
            print(f"Episode: {episode}, State: {next_state}, Action: {action}, Done: {done}")

            Q[state][action] = Q[state][action] + step_size * (reward + gamma*np.max(Q[next_state][:]) - Q[state][action])
            state = next_state
        
        returns.append(reward + gamma*np.max(Q[next_state][:]))

    return Q, returns