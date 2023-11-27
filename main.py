import env
import algorithms
import GORL


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
    env.render(mode='human')
    env.render(mode='matplotlib')


def train_agent(env, agent, num_epsiodes=1000):
    for episode in range(num_epsiodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_network(state, action, reward, next_state, done)
            total_reward = total_reward + reward
            state = next_state
            if done:
                break
        agent.update_target_network()
        agent.decay_epsilon()
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    # env = env.WarehouseEnvironment(grid_size=4, num_materials=2, num_obstacles=2)
    # visualize_environment(env)
    # MC = algorithms.monte_carlo(env, num_episodes=5)
    # visualize_environment(env)
    # print("Q-values after Monte Carlo:")
    # print(MC)
    env = env.WarehouseEnvironment(grid_size=5, num_materials=3, num_obstacles=2)
    num_materials = 3
    input_size = 4 + 2 * num_materials
    agent = GORL.DQNAgent(state_size= input_size, action_size=env.action_space.n)
    train_agent(env, agent)
