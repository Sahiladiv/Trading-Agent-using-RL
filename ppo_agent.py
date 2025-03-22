import gym
import numpy as np
import pandas as pd
import yfinance as yf
from gym.spaces import Box, Discrete
from gym import Env
import torch.nn as nn

class PPO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value