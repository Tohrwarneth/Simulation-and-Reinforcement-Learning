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
import numpy as np


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
        for _ in range(num_of_episodes):
            local_episode_buffer = []
            sim = simulation.Simulation(False, True)
            sim.run(local_episode_buffer)

            for index, pairing in enumerate(reversed(local_episode_buffer)):
                if index == 0:
                    avg_waiting = sim.finalAvgWaitingTime[len(sim.finalAvgWaitingTime)-1]
                    remaining = sim.personManager.get_remaining_people()
                self.__finalEpisodeBuffer.append(pairing + (avg_waiting, remaining,))
            utils.Clock.reset()

    def GetTrainingTensorsEntropy(self, percentage):
        '''Gets the training tensors for the best percenatage episode parings. 
        Result is state tensor and decision tensor.'''

        sortedEpisodes = sorted(self.__finalEpisodeBuffer, key=lambda entry: entry[2], reverse=True)
        elementsToTake = int(len(sortedEpisodes) * percentage)
        sortedEpisodes = sortedEpisodes[:elementsToTake]

        inputTensor = torch.zeros((elementsToTake, NetCoder.stateWidth), dtype=torch.float32)
        decisionTensor = torch.zeros(elementsToTake, dtype=torch.long)
        for i, (state, decision, avg_waiting, remaining) in enumerate(sortedEpisodes):
            NetCoder.encode_in_tensor(inputTensor, i, state)
            decisionTensor[i] = 1 if decision else 0

        return inputTensor, decisionTensor

    def get_training_tensors(self):
        '''
            Generates a training batch from the episodic buffer.
            batch_size: The amount of entries that are contained in the episodic buffer.
            
            return: tuple of input_tensor for the encided game state, a long tensor with the decisions, and a tensor with
            the associated state values.
        '''
        batch_size = len(self.__finalEpisodeBuffer)
        input_tensor = torch.zeros((batch_size, NetCoder.stateWidth), dtype=torch.float32)
        decision_tensor0 = torch.zeros(batch_size, dtype=torch.long)
        decision_tensor1 = torch.zeros(batch_size, dtype=torch.long)
        decision_tensor2 = torch.zeros(batch_size, dtype=torch.long)
        value_tensor = torch.zeros(batch_size, dtype=torch.float32)
        remaining_tensor = torch.zeros(batch_size, dtype=torch.float32)
        for i, (state, decision, avg_waiting, remaining) in enumerate(self.__finalEpisodeBuffer):
            NetCoder.encode_in_tensor(input_tensor, i, state)
            decision_tensor0[i] = decision[0]
            decision_tensor1[i] = decision[1]
            decision_tensor2[i] = decision[2]
            value_tensor[i] = avg_waiting
            remaining_tensor[i] = remaining_tensor

        return input_tensor, decision_tensor0, decision_tensor1, decision_tensor2, value_tensor, remaining_tensor

    # def get_input_tensor_numpy(self):
    #     '''
    #         Gets an input tensor as a numpy array where all the dice numbers are not normalized.
    #     '''
    #     batchSize = len(self.__finalEpisodeBuffer)
    #     result = np.zeros((batchSize, NetCoder.stateWidth))
    #     for i, (state, decision, score) in enumerate(self.__finalEpisodeBuffer):
    #         NetCoder.EncodeInTensorPlain(result, i, state)
    #
    #     return result
