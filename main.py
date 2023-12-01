import env
import algorithms
import gym


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


if __name__ == "__main__":

    env.register_env()
    env = gym.make('WarehouseEnv-v0',grid_size=4, num_materials=2, num_obstacles=2)
    
    # visualize_environment(env)
    # MC = algorithms.monte_carlo(env, num_episodes=100)
    # visualize_environment(env)
    # print("Q-values after Monte Carlo:")
    # print(MC)


    visualize_environment(env)
    MC = algorithms.q_learning(env=env,num_episodes=100,gamma=0.9,epsilon=0.1,step_size=0.5)
    visualize_environment(env)
    print("Q-values after Monte Carlo:")
    # print(MC)

