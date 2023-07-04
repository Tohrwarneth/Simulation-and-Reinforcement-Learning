import os

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

activation = nn.LeakyReLU()
# activation = nn.ReLU()


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=80, fc2_dims=256, fc3_dims=160, fc4_dims=40, chkpt_dir='model'):
        super(ActorNetwork, self).__init__()
        global activation

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            activation,
            nn.Linear(fc1_dims, fc2_dims),
            activation,
            nn.Linear(fc2_dims, fc3_dims),
            activation,
            nn.Linear(fc3_dims, fc4_dims),
            activation,
            nn.Linear(fc4_dims, fc1_dims),
            activation,
            nn.Linear(fc1_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        # self.__common = nn.Sequential(nn.Linear(57, fc1_dims), nn.ReLU(),
        #                               nn.Linear(fc1_dims, fc2_dims), nn.ReLU(),
        #                               nn.Linear(fc2_dims, 80), nn.ReLU())

        self.__common = nn.Sequential(nn.Linear(57, 80), nn.LeakyReLU(),
                                      nn.Linear(80, 256), nn.LeakyReLU(),
                                      nn.Linear(256, 160), nn.LeakyReLU(),
                                      nn.Linear(160, 80))

        self.__policy = nn.Sequential(
            nn.Linear(80, 60), nn.LeakyReLU(),
            nn.Linear(60, 40), nn.LeakyReLU(),
            nn.Linear(40, 28),
            nn.Softmax(dim=-1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')
        # TODO
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    # def forward(self, x):
    #     base = self.__common(x)
    #     logits = self.__policy(base)
    #     dist = Categorical(logits)
    #     return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    # One In-/Output: Value of a specific State
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='model'):
        super(CriticNetwork, self).__init__()

        global activation

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            activation,
            nn.Linear(fc1_dims, fc2_dims),
            activation,
            nn.Linear(fc2_dims, 1)
        )
        #
        # self.__common = nn.Sequential(nn.Linear(57, 80), nn.ReLU(),
        #                               nn.Linear(80, 160), nn.ReLU(),
        #                               nn.Linear(160, 80), nn.ReLU())

        self.__common = nn.Sequential(nn.Linear(57, 80), nn.LeakyReLU(),
                                      nn.Linear(80, 256), nn.LeakyReLU(),
                                      nn.Linear(256, 160), nn.LeakyReLU(),
                                      nn.Linear(160, 80))

        self.__value = nn.Sequential(
            nn.Linear(80, 60), nn.LeakyReLU(),
            nn.Linear(60, 40), nn.LeakyReLU(),
            nn.Linear(40, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # TODO
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.__value(self.__common(state))
        value = value.squeeze(-1)
        return value

    # def forward(self, state):
    #     value = self.critic(state)
    #     # value = value.squeeze(-1)
    #     return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
