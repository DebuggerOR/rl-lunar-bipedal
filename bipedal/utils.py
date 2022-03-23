import numpy as np


class GaussianNoise:
    def __init__(self, num_actions, mean=0.0):
        self.name = 'GaussianNoise'
        self.mean = mean
        self.size = num_actions

    def sample(self, std_dev=2.):
        x = np.random.normal(self.mean, std_dev, self.size)
        return x


class OUNoise:  # originally taken from: https://keras.io/examples/rl/ddpg_pendulum/
    def __init__(self, num_actions, mean=0.0, theta=1.5, dt=1e-2, x_initial=None):
        self.name = 'OUNoise'
        self.mean = mean * np.ones(num_actions)
        self.num_actions = num_actions
        self.theta = theta
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def sample(self, std_dev):
        std_dev = std_dev * np.ones(self.num_actions)
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, num_actions, seed, units_fc1=256):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_dim, units_fc1)
        self.linear3 = nn.Linear(units_fc1, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = torch.tanh(self.linear3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, num_actions, seed, units_fc1=256, units_fc2=256, units_fc3=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_dim + num_actions, units_fc1)
        self.linear2 = nn.Linear(units_fc1, units_fc2)
        self.linear3 = nn.Linear(units_fc2, units_fc3)
        self.linear4 = nn.Linear(units_fc3, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x