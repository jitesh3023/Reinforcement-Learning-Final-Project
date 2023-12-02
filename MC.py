import numpy as np

def monte_carlo_control(env, num_episodes=5, gamma=0.9, epsilon=0.1, epsilon_decay=0.99):
    Q = {}  # Dictionary to store state-action values
    
    def generate_episode(policy):
        episode = []
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            env.render(mode='matplotlib')  # Visualize the environment
        return episode

    def update_q_values(episode):
        return_estimate = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            state_tuple = tuple(state)
            return_estimate = gamma * return_estimate + reward
            if (state_tuple, action) not in Q:
                Q[(state_tuple, action)] = return_estimate
            else:
                Q[(state_tuple, action)] = (Q[(state_tuple, action)] + return_estimate) / 2

    def epsilon_greedy_policy(state):
        state_tuple = tuple(state)
        if np.random.rand() < epsilon:
            return env.action_space.sample()  # Explore
        else:
            return np.argmax([Q.get((state_tuple, a), 0) for a in range(env.action_space.n)])

    for _ in range(num_episodes):
        episode = generate_episode(epsilon_greedy_policy)
        update_q_values(episode)
        epsilon *= epsilon_decay  # Decay epsilon over episodes

    def get_optimal_policy():
        policy = {}
        for state in Q.keys():
            policy[state[0]] = np.argmax([Q.get((state[0], a), 0) for a in range(env.action_space.n)])
        return policy

    return get_optimal_policy()
