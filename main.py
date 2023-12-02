import env
import algorithms
import GORL
import MC


# env = env.WarehouseEnvironment(grid_size=10, num_materials=5, num_obstacles=7)
# print('Initial Environment: ')
# env.render(mode='human')
# env.render(mode='matplotlib')

# env.step(action=3) 
# print('Environment after taking a step: ')
# env.render(mode='human')
# env.render(mode='matplotlib')

# env.step(action=2) 
# print('Environment after taking a step: ')
# env.render(mode='human')
# env.render(mode='matplotlib')

def visualize_environment(environment):
    environment.render(mode='human')
    environment.render(mode='matplotlib')


def train_agent(env, agent, num_epsiodes=1000):
    for episode in range(num_epsiodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            #print(state.shape)
            agent.update_q_network(state, action, reward, next_state, done)
            total_reward = total_reward + reward
            state = next_state
            if done:
                break
        agent.update_target_network()
        agent.decay_epsilon()
        #print(state.shape)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    # OLD MC
   
    # env = env.WarehouseEnvironment(grid_size=10, num_materials=2, num_obstacles=2)
    # visualize_environment(env)
    # MC = algorithms.monte_carlo(env, num_episodes=1000)
    # visualize_environment(env)
    # print("V-values after Monte Carlo:")
    # #print(MC)

    # DQN

    env = env.WarehouseEnvironment(grid_size=3, num_materials=2, num_obstacles=2)
    input_size = 8
    agent = GORL.DQNAgent(state_size= input_size, action_size=env.action_space.n)
    train_agent(env, agent)

    # New MC

    # # Usage
    # env = env.WarehouseEnvironment(grid_size=3, num_materials=2, num_obstacles=2)
    # visualize_environment(env)
    #  # Create an instance of the Monte Carlo algorithm
    # mc_policy = MC.monte_carlo_control(env, num_episodes=5)

    # # Print the optimal policy
    # print("Optimal Policy:")
    # for state, action in mc_policy.items():
    #     print(f"State: {state}, Action: {action}")