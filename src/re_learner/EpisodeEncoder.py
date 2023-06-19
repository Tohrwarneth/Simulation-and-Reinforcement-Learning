# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:47:48 2020

@author: chris
"""
from __future__ import annotations
import simulation
import utils
from re_learner import NetCoder
import torch


class EpisodeEncoder:
    def __init__(self, decider):
        '''Runs the episode encoder with a decider handed over.'''
        self.__epsiodeList = []
        self.__decider = decider
        self.__finalEpisodeBuffer = []

    def generate_episodes(self, num_of_episodes):
        '''Flushes the internal buffer and generates new episodes. This is meant for on policy learning.
            numOfEpisdes : The number of episodes we want to generate internally.
        '''
        self.__finalEpisodeBuffer = []
        for i in range(num_of_episodes):
            local_episode_buffer = []
            sim = simulation.Simulation(False, True)
            sim.run(local_episode_buffer)

            avg_waiting = None
            remaining = None
            for index, pairing in enumerate(reversed(local_episode_buffer)):
                if index == 0:
                    avg_waiting = sim.finalAvgWaitingTime[len(sim.finalAvgWaitingTime) - 1]
                    remaining = sim.personManager.get_remaining_people()
                self.__finalEpisodeBuffer.append(pairing + (75, 0,))
                # Destination ist Wartezeit von 75 Takten und 0 Personen im Haus
            sim.reset()
            print(f'\rEpisode: {i}/{num_of_episodes}', end='')
        print(f'\rEpisode: {num_of_episodes}/{num_of_episodes}')

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

        for i, (state, decision, avg_waiting, remaining) in enumerate(self.__finalEpisodeBuffer):
            NetCoder.encode_in_tensor(input_tensor, i, state)
            decision_tensor[i] = decision[0]
            # decision_tensor[i, 1] = decision[1]
            # decision_tensor[i, 2] = decision[2]
            value_tensor[i] = avg_waiting
            # value_tensor[i, 1] = remaining

        return input_tensor, decision_tensor, value_tensor,
        # return input_tensor, decision_tensor0, decision_tensor1, decision_tensor2, value_tensor, remaining_tensor
