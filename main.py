import env


env = env.WarehouseEnvironment(grid_size=10, num_materials=5, num_obstacles=7)
print('Initial Environment: ')
#env.render(mode='human')
env.render(mode='matplotlib')

env.step(action=3) # 0=left,  1=right, 2=up, 3=down
print('Environment after taking a step: ')
env.render(mode='matplotlib')

env.step(action=2) # 0=left,  1=right, 2=up, 3=down
print('Environment after taking a step: ')
env.render(mode='matplotlib')