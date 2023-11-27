import numpy as np
import env

# first-visit on-policy MC
def monte_carlo(env, num_episodes, gamma=0.9):
    Q = np.zeros((np.prod(env.observation_space.nvec), env.action_space.n))
    N = np.zeros((np.prod(env.observation_space.nvec), env.action_space.n))
    
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
                action = np.argmax(Q[state])
            
            next_state, reward, done, _ = env.step(action)
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state
            print(f"Episode: {episode}, State: {state}, Action: {action}, Done: {done}")

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

    return Q
