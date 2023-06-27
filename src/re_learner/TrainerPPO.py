# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 08:49:44 2020

@author: chris
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
import simulation
import utils
from re_learner import Net, EpisodeEncoder

# BATCH_SIZE = 32
# TODO: wieder auf 64 stellen
BATCH_SIZE = 64

FACTOR_POLICY = 10.0
FACTOR_ENTROPY = 2.0


class TrainerPPO:
    modelFile = 'model/elevator_dice.dat'
    __device: torch.device

    def __init__(self):
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.__device = torch.device(dev)
        self.__net = simulation.Simulation.load_rl_model(self.modelFile, self.__device)
        self.__net.to(self.__device)
        self.__generator = EpisodeEncoder.EpisodeEncoder(self.__net, self.__device)
        self.__optimizer = torch.optim.Adam(self.__net.parameters())

    def __ApplyTrainingRoundSingle(self):
        # TODO: Reward vielleicht durch Satisfactory anreichern. Also Bonus pro Person/Aktion für Einsteigen (+10),
        #       Aussteigen am Ziel (+20), Am Ende nicht zuhause (-120, 4*Einstieg + 4*Ausstieg)
        #
        # TODO: Wartezeit die aktuelle nehmen. Ist die am Ende 0, ist keiner mehr im Haus oder
        #       Unwahrscheinlich: Alle sind gleichzeitig auf ihre Etage und kommen erst im nächsten Takt raus
        self.__generator.generate_episodes(BATCH_SIZE)
        self.__optimizer.zero_grad()
        inputTensor, decisionTensor, destinationValue, rewardTensor = self.__generator.get_training_tensors()
        logits, value = self.__net(inputTensor)
        # logits = Ausgangsschicht vor Softmax
        # value ist G
        lossValue = F.mse_loss(value, destinationValue)
        # mean squared error

        # print("Loss Value " + str(lossValue))

        # TODO: Advantage ist Reward Funktion
        advantage = destinationValue - value.detach()
        # TODO: am Ende tangus über hau mich blau einsetzen und deshalb zwischen -1 und 1
        # advantage_remaining = remaining_value - value.detach()

        probs = F.softmax(logits, dim=1)  # Die einzelnen Handlungsmöglichkeiten

        playedProbs = probs[range(inputTensor.size()[0]), decisionTensor]
        logProbs = F.log_softmax(logits, dim=1)
        rValue = playedProbs / playedProbs.detach()
        #  print("R -value " + str(rValue))

        innerValue = torch.min(advantage * rValue, advantage * torch.clamp(rValue, 0.8, 1.2))

        lossPolicy = -innerValue.mean()
        # print("Loss Policy " + str(lossPolicy))
        entropy = torch.sum(logProbs * probs, dim=1)
        lossEntropy = - entropy.mean() * rewardTensor.mean()
        print("Mean Reward " + str(rewardTensor.mean().item()))
        print("Loss Entropy " + str(lossEntropy))
        totalLoss = lossValue + FACTOR_POLICY * lossPolicy + FACTOR_ENTROPY * lossEntropy
        print("Total Loss " + str(totalLoss))

        # TODO: Reward fkt in total loss (als Tensor?)
        #       next_state, reward = env.step(action)
        #       loss = -m.log_prob(action) * reward
        #       loss.backward()

        totalLoss.backward()
        self.__optimizer.step()

    def __TestNet(self, one_run=False):
        if one_run:
            runs = 1
        else:
            runs = 25
        # runs = 1000
        summe_waiting_time = 0.0
        summe_remaining = 0.0
        summe_reward = 0.0
        print(f'Teste Netz: 0 / {runs}', end='')
        for i in range(runs):
            print(f'\rTeste Netz: {i} / {runs}', end='')
            simulator = simulation.Simulation(show_gui=False, rl_decider=True)
            simulator.run()
            simulator.reset()
            remaining = simulator.personManager.get_remaining_people()
            summe_remaining += remaining
            summe_reward += np.mean(simulator.rewardList, axis=0)
            avg_waiting_time = simulator.avgWaitingTime[len(simulator.avgWaitingTime) - 1]
            summe_waiting_time += avg_waiting_time

        avg_remaining = summe_remaining / runs
        avg_waiting = summe_waiting_time / runs
        avg_reward = summe_reward / runs
        print(
            f"\rTest\tAverage Waiting Time: {avg_waiting:.2f}\tAverage Remaining: {avg_remaining:.2f}\t"
            f"Average Reward: {avg_reward:.2f}")
        return avg_remaining, avg_waiting, avg_reward

    def TrainNetwork(self):
        _, best_avg_waiting, best_avg_reward = self.__TestNet(True)
        i = 0
        while True:
            print(15 * '-')
            print(f'Epoche {i}:')
            print(5 * '-')
            for _ in range(1):
                # for i in range(10):
                self.__ApplyTrainingRoundSingle()
            avg_remaining, avg_waiting, reward = self.__TestNet()
            if reward >= best_avg_reward:
                best_avg_reward = reward
                os.makedirs(os.path.dirname(self.modelFile), exist_ok=True)
                torch.save(self.__net, self.modelFile)
                print(
                    f'Save model with avg. waiting time = {avg_waiting:.2f}, {avg_remaining} avg. remaining people '
                    f'and avg. reward = {reward:.2f}')
                print(5 * '-')
                if avg_remaining == 0:
                    break
            i += 1


if __name__ == "__main__":
    trainer = TrainerPPO()
    trainer.TrainNetwork()
