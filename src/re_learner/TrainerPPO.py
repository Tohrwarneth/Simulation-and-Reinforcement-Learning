"""
Class created by Prof. C. Lürig
modified from Zombie Dice to Elevator Simulation
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F

import plotter
import simulation
from re_learner import EpisodeEncoder

BATCH_SIZE = 64
BATCH_SIZE = 1

FACTOR_POLICY = 10.0
FACTOR_ENTROPY = 2.0


class TrainerPPO:
    modelFile = 'model/elevator_dice.dat'

    def __init__(self):
        self.__net = simulation.Simulation.load_rl_model(self.modelFile)
        self.__generator = EpisodeEncoder.EpisodeEncoder(self.__net)
        self.__optimizer = torch.optim.Adam(self.__net.parameters())

    def __ApplyTrainingRoundSingle(self):
        self.__generator.generate_episodes(BATCH_SIZE)
        self.__optimizer.zero_grad()
        inputTensor, decisionTensor, destinationValue, rewardTensor = self.__generator.get_training_tensors()
        logits, value = self.__net(inputTensor)
        lossValue = F.mse_loss(value, destinationValue)

        advantage = (destinationValue - value.detach())

        probs = F.softmax(logits, dim=1)

        playedProbs = probs[range(inputTensor.size()[0]), decisionTensor]
        logProbs = F.log_softmax(logits, dim=1)
        rValue = playedProbs / playedProbs.detach()

        print(f"Mean Reward {rewardTensor.mean().item():.2f}")

        innerValue = torch.min(advantage * rValue, advantage * torch.clamp(rValue, 0.8, 1.2))

        lossPolicy = -innerValue.mean() * rewardTensor.mean()
        entropy = torch.sum(logProbs * probs, dim=1)
        lossEntropy = - entropy.mean()
        totalLoss = lossValue + FACTOR_POLICY * lossPolicy + FACTOR_ENTROPY * lossEntropy

        totalLoss.backward()
        self.__optimizer.step()

    def __TestNet(self) -> tuple[float, float, float]:
        '''
        Tests net in 10 runs
        :return: avg remaining persons, avg waiting time, avg reward
        '''
        runs = 10
        runs = 1
        sum_waiting_time = 0.0
        sum_remaining = 0.0
        sum_reward = 0.0
        print(f'Teste Netz: 0 / {runs}', end='')
        for i in range(runs):
            print(f'\rTeste Netz: {i} / {runs}', end='')
            simulator = simulation.Simulation(show_gui=False, rl_decider=True)
            simulator.run()
            remaining = simulator.personManager.get_remaining_people()
            sum_remaining += remaining
            sum_reward += sum(simulator.rewardList)
            avg_waiting_time = simulator.avgWaitingTime[len(simulator.avgWaitingTime) - 1]
            sum_waiting_time += avg_waiting_time

        avg_remaining = sum_remaining / runs
        avg_waiting = sum_waiting_time / runs
        avg_reward = sum_reward / runs
        print(
            f"\rTest\tAverage Waiting Time: {avg_waiting:.2f}\tAverage Remaining: {avg_remaining:.2f}\t"
            f"Average Reward: {avg_reward:.2f}")
        return avg_remaining, avg_waiting, avg_reward

    def TrainNetwork(self) -> None:
        '''
        Train network of dice implementation
        '''
        _, best_avg_waiting, best_avg_reward = self.__TestNet()
        i = 0
        self.avg_reward_list = list()
        while True:
            print(15 * '-')
            print(f'Epoche {i}:')
            print(5 * '-')
            self.__ApplyTrainingRoundSingle()
            avg_remaining, avg_waiting, avg_reward = self.__TestNet()
            self.avg_reward_list.append(avg_reward)

            if avg_reward >= best_avg_reward:
                # save best model
                best_avg_reward = avg_reward
                os.makedirs(os.path.dirname(self.modelFile), exist_ok=True)
                torch.save(self.__net, self.modelFile)
                print(
                    f'Save model with avg. waiting time = {avg_waiting:.2f}, {avg_remaining} avg. remaining people '
                    f'and avg. reward = {avg_reward:.2f}')
                print(5 * '-')
                if avg_remaining == 0:
                    # target reached
                    break
            i += 1
            x = [i + 1 for i in range(len(self.avg_reward_list))]
            plotter.plot_learning_curve(x, self.avg_reward_list, 'images/train_dice.png')
