# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:47:48 2020

@author: chris
"""
from __future__ import annotations

import numpy as np

import simulation
import utils
from re_learner import NetCoder
import torch


class EpisodeEncoder:
    def __init__(self, decider, device):
        '''Runs the episode encoder with a decider handed over.'''
        self.__epsiodeList = []
        self.__decider = decider
        self.__device = device
        self.__finalEpisodeBuffer = []

    def generate_episodes(self, num_of_episodes):
        '''Flushes the internal buffer and generates new episodes. This is meant for on policy learning.
            numOfEpisdes : The number of episodes we want to generate internally.
        '''
        self.__finalEpisodeBuffer = []
        avg_waiting = 0.0
        remaining = utils.Conf.totalAmountPerson
        best_remaining = 0
        best_waiting = 0
        best_reward = 0
        best_episode = -1
        for i in range(num_of_episodes):
            print(f'\rEpisode: {i}/{num_of_episodes}\t', end='')
            local_episode_buffer = []
            sim = simulation.Simulation(False, True)
            sim.run(local_episode_buffer)

            for index, pairing in enumerate(reversed(local_episode_buffer)):
                self.__finalEpisodeBuffer.append(pairing + (75, 0,))
                # Destination ist Wartezeit von 75 Takten und 0 Personen im Haus
            avg_waiting = sim.avgWaitingTime[len(sim.avgWaitingTime) - 1]
            remaining = sim.personManager.get_remaining_people()
            reward = np.mean(sim.rewardList, axis=0)
            if best_episode == -1:
                best_remaining = remaining
                best_reward = reward
                best_waiting = avg_waiting
                best_episode = i + 1
            else:
                best_reward = min(best_reward, reward)
                best_remaining = min(best_remaining, remaining)
                best_waiting = min(best_waiting, avg_waiting)
            sim.reset()
        print(
            f'\rEpisode: {num_of_episodes}/{num_of_episodes}\tBeste Episode: {best_episode}\t Waiting Time = {best_waiting:.2f}\t'
            f'Remaining = {best_remaining}/{utils.Conf.totalAmountPerson}\t Reward = {best_reward:.2f}')

    def get_training_tensors(self):
        '''
            Generates a training batch from the episodic buffer.
            batch_size: The amount of entries that are contained in the episodic buffer.
            
            return: tuple of input_tensor for the encided game state, a long tensor with the decisions, and a tensor with
            the associated state values.
        '''
        batch_size = len(self.__finalEpisodeBuffer)
        input_tensor = torch.zeros((batch_size, NetCoder.stateWidth), dtype=torch.float32)
        decision_tensor = torch.zeros(batch_size, dtype=torch.long)
        value_tensor = torch.zeros(batch_size, dtype=torch.float32)
        reward_tensor = torch.zeros(batch_size, dtype=torch.float32)

        loading_dots = 1
        print('Build trainings tensor', end='')
        for i, (state, decision, reward, avg_waiting, remaining) in enumerate(self.__finalEpisodeBuffer):
            NetCoder.encode_in_tensor(input_tensor, i, state)
            decision_tensor[i] = NetCoder.states_to_decision(decision)
            value_tensor[i] = avg_waiting
            reward_tensor[i] = reward
            # TODO: Ist Destination Value und wird für die Rewardfkt genutzt. Oder ist das der Reward?
            # value_tensor[i, 1] = remaining
            print(f'\rBuild trainings tensor {loading_dots * "."}', end='')
            loading_dots += 1
            if loading_dots > 3:
                loading_dots = 1
        print()
        input_tensor.to(self.__device)
        decision_tensor.to(self.__device)
        value_tensor.to(self.__device)
        reward_tensor.to(self.__device)
        return input_tensor, decision_tensor, value_tensor, reward_tensor
