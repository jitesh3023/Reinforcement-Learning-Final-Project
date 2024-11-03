import gym
from gym import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from env import WarehouseEnvironment

# Define the Q-network architecture for goal-oriented DQN
class GoalDQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(GoalDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        print(input_size)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, state):
        if len(state.shape) == 3:
            state = state.squeeze(0).squeeze(0)
        print("Input shape before fc1:", state.shape)
        x = torch.relu(self.fc1(state))
        print("After fc1 shape:", x.shape)
        x = torch.relu(self.fc2(x))
        print("After fc2 shape:", x.shape)
        x = self.fc3(x)
        print("Output shape:", x.shape)
        return x

# Define the GoalDQNAgent
class GoalDQNAgent:
    def __init__(self, state_size, action_size):
        self.q_network = GoalDQNNetwork(state_size, action_size)
        self.target_network = GoalDQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.state_size = state_size
        self.action_size = action_size
    def update_q_network(self, batch, gamma=0.99):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.q_network(states)
        next_q_values = self.target_network(next_states).detach()

        target_values = rewards + gamma * (1 - dones) * next_q_values.unsqueeze(0).max(dim=1).values

        selected_q_values = q_values[range(len(q_values)), actions.view(-1).long()]

        loss = nn.MSELoss()(selected_q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return torch.tensor(random.randint(0, self.action_size - 1))
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()



    def train(self, num_episodes, max_steps=100):
        gamma = 0.99
        update_frequency = 5
        target_update_frequency = 10

        for episode in range(num_episodes):
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            total_reward = 0

            for step in range(max_steps):
                epsilon = max(0.1, 1.0 - episode / 500)  # Epsilon-greedy exploration
                action = self.select_action(state, epsilon)

                next_state, reward, done, _ = env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                transition = (state, action, reward, next_state, done)
                self.update_q_network([transition], gamma)

                state = next_state
                total_reward += reward

                if step % target_update_frequency == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

                if done:
                    break

            if episode % update_frequency == 0:
                print(f"Episode: {episode}, Total Reward: {total_reward}")


# Training the agent
env = WarehouseEnvironment(grid_size=5, num_materials=3, num_obstacles=2)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = GoalDQNAgent(state_size=10, action_size=4)

# Train the agent
agent.train(num_episodes=500)

# Test the trained agent
state = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
for _ in range(10):
    action = agent.select_action(state, epsilon=0.0)
    state, _, _, _ = env.step(action.item())
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    env.render(mode='matplotlib')