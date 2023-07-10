"""
Class created by Prof. C. Lürig
modified from Zombie Dice to Elevator Simulation
"""

import torch
import torch.nn as nn
from re_learner import NetCoder


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        activation = nn.LeakyReLU()
        self.__common = nn.Sequential(nn.Linear(57, 80), activation,
                                      nn.Linear(80, 160), activation,
                                      nn.Linear(160, 80), activation)

        self.__value = nn.Sequential(
            nn.Linear(80, 40), activation,
            nn.Linear(40, 1), activation)

        self.__policy = nn.Sequential(
            nn.Linear(80, 40), activation,
            nn.Linear(40, 28))

    def forward(self, x):
        base = self.__common(x)
        value = self.__value(base) * 13.0
        # These will be all 1 element arrays.
        value = value.squeeze(-1)
        logits = self.__policy(base)
        # Policy is returned in logits
        return logits, value

    def decide_for_action(self, game_state) -> int:
        '''

        :param game_state: current game state
        :return: Decision index
        '''
        in_tensor = torch.zeros((1, NetCoder.stateWidth), dtype=torch.float32)
        NetCoder.encode_in_tensor(in_tensor, 0, game_state)
        with torch.no_grad():
            logits, value = self(in_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
        return action[0].item()


if __name__ == "__main__":
    print(Net())
