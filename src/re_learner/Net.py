# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 07:51:46 2020

@author: chris
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

        # We know that value >= 0
        self.__value = nn.Sequential(
            nn.Linear(80, 40), activation,
            nn.Linear(40, 1), activation)
        # Gesamt Situation

        self.__policy = nn.Sequential(
            nn.Linear(80, 40), activation,
            nn.Linear(40, 2))
        # TODO: aktuell Zieletage. Vlt zu 3 Handlungen (Warten, Hoch, Runter) statt Etagenummer für jeden Aufzug
        #       Mache eine Entscheidung aus 27 Möglichkeiten. Also zwischen 0-26 und das Kreuzprodukt der Handlungen
        #       (27) Durchnummerieren (x mod 27)
        #       Dadurch keine Erweiterungen
        # Handlung

    def forward(self, x):
        base = self.__common(x)
        value = self.__value(base) * 13.0
        # TODO: Warum mal 13?
        # These will be all 1 element arrays.
        value = value.squeeze(-1)
        logits = self.__policy(base)
        # Policy is returned in logits
        return logits, value

    def decide_for_action(self, game_state) -> int:
        in_tensor = torch.zeros((1, NetCoder.stateWidth), dtype=torch.float32)
        NetCoder.encode_in_tensor(in_tensor, 0, game_state)
        with torch.no_grad():
            logits, value = self(in_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs = probs[0].data.cpu().numpy()

        return probs[1]
