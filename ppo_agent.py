import torch
import torch.nn as nn

class PPO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Linear(256, 1)

        self.apply(self._init_weights)

    def forward(self, state):
        shared_output = self.shared(state)
        action_probs = self.actor(shared_output)
        value = self.critic(shared_output)
        return action_probs, value

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
