import numpy as np
import env
import random

def argmax(arr, action_space_size):
    max_value = np.max(arr)
    valid_indices = [i for i, val in enumerate(arr) if val == max_value and i < action_space_size]
    #print(valid_indices)
    max_ = random.choice(valid_indices)
    return max_



# first-visit on-policy MC
def monte_carlo(env, num_episodes, gamma=0.99):
    total_states = np.prod(env.observation_space.nvec)
    # V = np.zeros((total_states, env.action_space.n))
    # N = np.zeros((total_states, env.action_space.n))
    V = np.zeros((total_states))
    N = np.zeros((total_states))
    state = env.reset()  # setting the robot/agent state to start

    for episode in range(num_episodes):
        episode_states = []
        episode_actions = []
        episode_rewards = []
        #print(state)
        
        # Generating the episode with a policy
        done = False
        
        while not done:
            if np.random.rand() < 0.1:
                action = np.random.choice(env.action_space.n)
                #print(action)
            else:
                action = argmax(V[state], env.action_space.n)
                #action = 0
            #action = np.random.choice(env.action_space.n)
            #print(env.action_space)
            #print(env.action_space.n)
            #print(env.action_space.sample())
            #env.render(mode='matplotlib')  # Visualize the environment            
            next_state, reward, done, _ = env.step(action)
            
            #print(V[index_t])
            #print(next_state)
            episode_states. append(state)
            episode_actions.append(action)  
            episode_rewards.append(reward)
            state = next_state
            #print(episode_rewards)
            #print(f"Episode: {episode}, State: {state}, Action: {action}, Done: {done}")

        G = 0
        visited_states = []

        for t in range(len(episode_states) - 1, -1, -1):
            state_t = tuple((episode_states[t][0], episode_states[t][1]))  # Convert to tuple with a single element
            #state_t = (tuple([state]),  )
            #print(state_t)
            action_t = episode_actions[t]
            reward_t = episode_rewards[t]
            G = gamma * G + reward_t
            

            # Unless S_t appears in visited_states:
            # Unless S_t appears in visited_states:
            # Unless S_t appears in visited_states:
            if not any(np.array_equal(state_t, visited_state) for visited_state in visited_states):
                visited_states.append(state_t)
                index_t = hash(state_t) % len(N)
                N[index_t] += 1
                #print(N[index_t])
                #print(G)
                V[index_t] += (1 / N[index_t]) * (G - V[index_t])
                

                
                #  print(f"State: {state_t}, Action: {action_t}, Reward: {reward_t}, Q-values: {Q[state_t]}")
                #print(Q)
                #print("I am here")
                #print(N)
                #print(V[index_t])
    return V
