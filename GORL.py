import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Defining Neural Network
class QNetwork(nn.Module): # Using pytorch's nn.Module we will define NN
    def __init__(self, input_size, output_size):
        # Here input_size - represents the dimensionality or size of the input to the NN. In other words it correspondes to the number of features or variables that descrive the state of the environment
        # output_size - represets the number of neurons in the output layer of theNN. Like number of possible actiosn the agent can take in the envrionment
        super(QNetwork, self).__init__()
        self.fully_connected_layer_1 = nn.Linear(input_size, 64) #input_features - output-features
        self.fully_connected_layer_2 = nn.Linear(64, 32)
        self.fully_connected_layer_3 = nn.Linear(32, output_size)
    def forward(self, x): # doing forward propogation in the NN
        x = torch.relu(self.fully_connected_layer_1(x)) # applyting activation function on FC1
        x = torch.relu(self.fully_connected_layer_2(x))
        return self.fully_connected_layer_3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        #state_size - represents the dimensionality of the state space in the RL env, basically defines number of features or variable that describe the current state
        #action_size - represents the size of action space
        #learning rate - controls hte step size during optimization
        #gamma - discount factor
        #epsilon_start - represents the initial value of the exploration-exploitation, initially set to 1 meaning full exploration.
        #epsilon_min - least value possible for epsilon
        #epsilon_decay - rate at which epsilon decays over time
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min= epsilon_min
        self.epsilon_decay = epsilon_decay
        # Q-network Initialization 
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size) #Creating a copy instance of Q network
        # Target Netwrok Initialization
        self.target_network.load_state_dict(self.q_network.state_dict())# this will initialize target Q network with the same parameters as the main Q-network
        self.target_network.eval() 
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        if np.random.rand() < self.epsilon: # exploration
            return np.random.choice(self.action_size)
        else: # exploitation
            state = torch.FloatTensor(state).unsqueeze(0) # converting the input state into Pytorch tensor, plus adding a batch dimension
            with torch.no_grad():# disabling gradient computation during the forward pass
                q_values = self.q_network(state) # computing Q-values for each action
            return torch.argmax(q_values).item() # highest Q value selected for exploitation
        
    def update_q_network(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.tensor([action])
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)
        q_value =self.q_network(state).gather(1, action.unsqueeze(1))
        with torch.no_grad():
            next_q_value = self.target_network(next_state).max(1)[0].unsqueeze(1)
            target = reward + (1-done) * self.gamma*next_q_value
        loss = nn.MSELoss()(q_value, target) # Calculates mean_squared error between the predicted value and the target value
        self. optimizer.zero_grad() # clearing gradients before computing new gradients
        loss.backward() # Computes the gradients of loss wrt parameters of Q-network
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        
           
