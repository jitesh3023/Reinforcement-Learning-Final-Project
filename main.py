import env
import algorithms
import gym
import matplotlib.pyplot as plt
import numpy as np


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

def get_action(action):
    if action == 3:
        return "R"
    
    if action == 1:
        return "D"
    
    if action == 2:
        return "L"
    
    if action == 0:
        return "U"

def visualize_environment(environment):
    env.render(mode='human')
    env.render(mode='matplotlib')


if __name__ == "__main__":

    env.register_env()
    env = gym.make('WarehouseEnv-v0',grid_size=5, num_materials=2, num_obstacles=1)
    
    # visualize_environment(env)
    # MC, returns_plot = algorithms.monte_carlo(env, num_episodes=10000)
    #visualize_environment(env)
    #print("Q-values after Monte Carlo:")
    # print(MC)

    # Print the optimal policy
    # print("Optimal Policy:")
    # for state, actions in MC.items():
    #     print(f"State: {state}, Action: {get_action(np.argmax(actions))}")

    # robot_position = (0,0)
    # print("Path")

    # while robot_position != (4,4):
    #     action = algorithms.argmax(MC[robot_position])
    #     print(f"State: {robot_position} Action: {get_action(action)}")
        
    #     if action == 0: #Go up
    #         new_position=(robot_position[0]-1, robot_position[1])
    #     elif action == 1: #Go down
    #         new_position=(robot_position[0]+1, robot_position[1])
    #     elif action == 2: #Go left
    #         new_position=(robot_position[0], robot_position[1]-1)
    #     elif action ==3: #Go right
    #         new_position=(robot_position[0], robot_position[1]+1)

    #     robot_position = new_position
    # plt.plot(returns_plot)
    # plt.show()

    visualize_environment(env)
    MC, return_plot = algorithms.q_learning(env=env,num_episodes=100000,gamma=0.9,epsilon=0.1,step_size=0.5)
    # visualize_environment(env)
    print("Q-values after Monte Carlo:")
    #print(MC)
    # plt.plot(return_plot)
    # plt.show()
        # Print the optimal policy
    # print("Optimal Policy:")
    # for state, actions in MC.items():
    #     print(f"State: {state}, Action: {get_action(np.argmax(actions))}")

    for i in range(env.grid_size):
        for j in range(env.grid_size):
            position = (i,j)
            print(get_action(algorithms.argmax(MC[position])), end=' ')
        print()

    # robot_position = (0,0)
    # print("Path")

    # while robot_position != (4,4):
    #     action = algorithms.argmax(MC[robot_position])
    #     print(f"State: {robot_position} Action: {get_action(action)}")
        
    #     if action == 0: #Go up
    #         new_position=(robot_position[0]-1, robot_position[1])
    #     elif action == 1: #Go down
    #         new_position=(robot_position[0]+1, robot_position[1])
    #     elif action == 2: #Go left
    #         new_position=(robot_position[0], robot_position[1]-1)
    #     elif action ==3: #Go right
    #         new_position=(robot_position[0], robot_position[1]+1)

    #     robot_position = new_position