import torch
from torch import nn
import random

class ClashModel(nn.Module):
    def __init__(self, in_features, hidden_units, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=out_features)
        )

    
    def forward(self, x):
        x = self.layers(x)
        return x


class ClashAgent:
    def __init__(self, in_features, out_features):
        self.model = ClashModel(in_features=in_features,
                   hidden_units=10,
                   out_features=out_features)
        self.epsilon = 0.3
        self.gamma = 0.95


    def act(self, state):
        q_values = self.model(state)
        action_idx = torch.argmax(q_values)
        return action_idx
    

    def step(self):
        pass