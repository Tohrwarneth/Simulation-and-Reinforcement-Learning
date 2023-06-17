# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 08:49:44 2020

@author: chris
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import simulation
import utils
from re_learner import Net, EpisodeEncoder

BATCH_SIZE = 64

FACTOR_POLICY = 10.0
FACTOR_ENTROPY = 2.0


class TrainerPPO:
    modelFile = 'model/elevator.dat'

    def __init__(self):
        self.__net = Net.Net()
        self.__generator = EpisodeEncoder.EpisodeEncoder(self.__net)
        self.__optimizer = torch.optim.Adam(self.__net.parameters())

    def __ApplyTrainingRoundSingle(self):
        self.__generator.generate_episodes(BATCH_SIZE)
        self.__optimizer.zero_grad()
        inputTensor, decision_tensor0, decision_tensor1, decision_tensor2, waiting_tensor, remaining_value = self.__generator.get_training_tensors()
        logits, value = self.__net(inputTensor)
        lossValue = F.mse_loss(value, waiting_tensor)
        lossRemaining = F.mse_loss(value, remaining_value)

        print("Loss Value " + str(lossValue))
        advantage = waiting_tensor - value.detach()
        advantage_remaining = remaining_value - value.detach()

        probs = F.softmax(logits, dim=1)

        playedProbs = probs[range(inputTensor.size()[0]), decision_tensor0]
        logProbs = F.log_softmax(logits, dim=1)
        rValue = playedProbs / playedProbs.detach()
        #  print("R -value " + str(rValue))

        innerValue = torch.min(advantage * rValue, advantage * torch.clamp(rValue, 0.8, 1.2))

        lossPolicy = -innerValue.mean()
        # print("Loss Policy " + str(lossPolicy))
        entropy = torch.sum(logProbs * probs, dim=1)
        lossEntropy = - entropy.mean()
        # print("Loss Entropy " + str(lossEntropy))
        totalLoss = lossValue + FACTOR_POLICY * lossPolicy + FACTOR_ENTROPY * lossEntropy
        totalLoss.backward()
        self.__optimizer.step()

    def __TestNet(self):
        runs = 1000
        summe_waiting_time = 0.0
        summe_remaining = 0.0
        for _ in range(runs):
            simulator = simulation.Simulation(show_gui=False, rl_decider=True)
            simulator.run()
            utils.Clock.reset()
            remaining = simulator.personManager.get_remaining_people()
            summe_remaining += remaining
            avg_waiting_time = simulator.avgWaitingTime
            summe_waiting_time += avg_waiting_time

        avg_remaining = summe_waiting_time / runs
        avg_waiting = summe_waiting_time / runs
        print("Test Average Remaining: " + str(avg_remaining) + '| Test Average Waiting Time: ' + str(avg_waiting))
        return avg_remaining, avg_waiting

    def TrainNetwork(self):
        while True:
            for _ in range(10):
                self.__ApplyTrainingRoundSingle()
            avg_remaining, avg_waiting = self.__TestNet()
            if avg_remaining == 0:
                torch.save(self.__net, self.modelFile)
                break


if __name__ == "__main__":
    trainer = TrainerPPO()
    trainer.TrainNetwork()
