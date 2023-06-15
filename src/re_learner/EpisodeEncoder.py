# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:47:48 2020

@author: chris
"""

import simulation
import NetCoder
import torch
import numpy as np

class EpisodeEncoder:
    def __init__(self, decider):
        '''Runs the episode encoder with a decider handed over.'''
        self.__epsiodeList = []
        self.__sim = simulation.Simulation()
        self.__decider = decider
        self.__finalEpisodeBuffer = []
        
    def GenerateEpisodes(self, numOfEpisodes):
        '''Flushes the internal buffer and generates new episodes. This is meant for on policy learning.
            numOfEpisdes : The number of episodes we want to generate internally.
        '''
        self.__finalEpisodeBuffer = []
        for _ in range(numOfEpisodes):
            localEpisodeBuffer = []
            self.__sim.Reset()
            isFinished = False
            while not isFinished:
                startState = self.__sim.GetState()
                goOn = self.__decider.DecideForAction(startState)
                isFinished, score = self.__sim.ApplyDecision(goOn)
                localEpisodeBuffer.append( (startState,  goOn))
            for pairing in reversed(localEpisodeBuffer):
                self.__finalEpisodeBuffer.append(pairing + (score,))
                
                
    def GetTrainingTensorsEntropy(self, percentage):
        '''Gets the training tensors for the best percenatage episode parings. 
        Result is state tensor and decision tensor.'''
        
        sortedEpisodes = sorted(self.__finalEpisodeBuffer, key = lambda entry : entry[2], reverse=True)
        elementsToTake = int(len(sortedEpisodes) * percentage)
        sortedEpisodes = sortedEpisodes[:elementsToTake]
        
        inputTensor = torch.zeros((elementsToTake, NetCoder.StateWidth), dtype=torch.float32)
        decisionTensor = torch.zeros(elementsToTake, dtype=torch.long)
        for i, (state, decision, score) in enumerate(sortedEpisodes):
            NetCoder.EncodeInTensor(inputTensor, i, state)
            decisionTensor[i] = 1 if decision else 0
            
        return inputTensor, decisionTensor
                
    
    
    def GetTrainingTensors(self):
        '''
            Generates a training batch from the episodic buffer.
            batchSize: The amount of entries that are contained in the episodic buffer.
            
            return: tuple of inputTensor for the encided game state, a long tensor with the decisions, and a tensor with
            the associated state values.
        '''
        batchSize = len(self.__finalEpisodeBuffer)
        inputTensor = torch.zeros((batchSize, NetCoder.StateWidth), dtype=torch.float32)
        decisionTensor = torch.zeros(batchSize, dtype=torch.long)
        valueTensor = torch.zeros(batchSize, dtype=torch.float32)
        for i, (state, decision, score) in enumerate(self.__finalEpisodeBuffer):
            NetCoder.EncodeInTensor(inputTensor, i, state)
            decisionTensor[i] = 1 if decision else 0
            valueTensor[i] = score
            
        return inputTensor, decisionTensor, valueTensor
    
    
    def GetInputTensorNumpy(self):
        '''
            Gets an input tensor as a numpy array where all the dice numbers are not normalized.
        '''
        batchSize = len(self.__finalEpisodeBuffer)
        result = np.zeros((batchSize, NetCoder.StateWidth))
        for i, (state, decision, score) in enumerate(self.__finalEpisodeBuffer):
           NetCoder.EncodeInTensorPlain(result, i, state)
           
        return result
        
        
                
            
