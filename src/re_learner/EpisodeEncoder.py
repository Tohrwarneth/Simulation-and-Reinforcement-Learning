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
        avg_waiting = 0.0
        remaining = utils.Conf.totalAmountPerson
        for i in range(num_of_episodes):
            print(f'\rEpisode: {i}/{num_of_episodes}\tavg. waiting = {avg_waiting}\t'
                  f'remaining = {remaining}/{utils.Conf.totalAmountPerson}', end='')
            local_episode_buffer = []
            sim = simulation.Simulation(False, True)
            sim.run(local_episode_buffer)

            for index, pairing in enumerate(reversed(local_episode_buffer)):
                if index == 0:
                    avg_waiting = sim.avgWaitingTime[len(sim.avgWaitingTime) - 1]
                    remaining = sim.personManager.get_remaining_people()
                self.__finalEpisodeBuffer.append(pairing + (75, 0,))
                # Destination ist Wartezeit von 75 Takten und 0 Personen im Haus
            sim.reset()
        print(f'\rEpisode: {num_of_episodes}/{num_of_episodes}\tavg. waiting = {avg_waiting}\t'
              f'remaining = {remaining}/{utils.Conf.totalAmountPerson}')

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

        loading_dots = 1
        print('Build trainings tensor', end='')
        for i, (state, decision, avg_waiting, remaining) in enumerate(self.__finalEpisodeBuffer):
            NetCoder.encode_in_tensor(input_tensor, i, state)
            decision_tensor[i] = NetCoder.states_to_decision(decision)
            value_tensor[i] = avg_waiting
            # TODO: Ist Destination Value und wird für die Rewardfkt genutzt. Oder ist das der Reward?
            # value_tensor[i, 1] = remaining
            print(f'\rBuild trainings tensor {loading_dots * "."}', end='')
            loading_dots += 1
            if loading_dots > 3:
                loading_dots = 1
        print()
        return input_tensor, decision_tensor, value_tensor,
        # return input_tensor, decision_tensor0, decision_tensor1, decision_tensor2, value_tensor, remaining_tensor
