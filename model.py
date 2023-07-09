'''
Creating the model, will be a deep Q network, either Dueling deep Q network or deep Q network

Simple linear neural network used

This neural network predicts the Q value for each possible action

'''

import torch
from torch import nn
import torch.nn.functional as F

class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, network_type='DQN', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network_type = network_type
        self.layer1 = nn.Linear(input_dim,64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 512) 
        self.layer5 = nn.Linear(512, 512)
        # # Dueling DQN predicts both the value of the state and the advantage of each possible action
        # # Best action should have advantage of 0
        ## Outputs are combined to generate the Q values

        if network_type == 'DuelingDQN':
            self.state_values = nn.Linear(512,1)
            self.advantages = nn.Linear(512, output_dim)
        else:
            self.output = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu6(self.layer1(x))
        x = F.relu6(self.layer2(x))
        x = F.relu6(self.layer3(x))
        x = F.relu6(self.layer4(x))
        x = F.relu6(self.layer5(x))
        if self.network_type == 'DuelingDQN':
            state_values = self.state_values(x)
            advantages = self.advantages(x)
            output = state_values + (advantages - torch.max((advantages), dim=1, keepdim=True)[0])
            return output
        else:
            return self.output(x)
        

# class DQN(nn.Module):
#     def __init__(self, input_dim, output_dim, network_type='DDQN', *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.network_type = network_type
#         self.layer1 = nn.Linear(input_dim, 128)
#         self.layer2 = nn.Linear(128, 256)
#         self.layer3 = nn.Linear(256, 256)
#         # # Dueling DQN predicts both the value of the state and the advantage of each possible action
#         # # Best action should have advantage of 0
#         ## Outputs are combined to generate the Q values

#         if network_type == 'DuelingDDQN':
#             self.state_values = nn.Linear(256,1)
#             self.advantages = nn.Linear(256, output_dim)
#         else:
#             self.output = nn.Linear(256, output_dim)
  
#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         x = F.relu(self.layer3(x))
#         if self.network_type == 'DuelingDDQN':
#             state_values = self.state_values(x)
#             advantages = self.advantages(x)
#             output = state_values + (advantages - torch.max((advantages), dim=1, keepdim=True)[0])
#             return output
#         else:
#             return self.output(x)
        
