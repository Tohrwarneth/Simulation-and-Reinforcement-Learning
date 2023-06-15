# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 08:49:44 2020

@author: chris
"""


import Net
import EpisodeEncoder
import torch
import torch.nn.functional as F
import simulation

BATCH_SIZE = 64

FACTOR_POLICY = 10.0
FACTOR_ENTROPY =   2.0

class TrainerPPO:
    def __init__(self):
        self.__net = Net.Net()
        self.__generator = EpisodeEncoder.EpisodeEncoder(self.__net)
        self.__optimizer = torch.optim.Adam(self.__net.parameters())
        
   
    
        
    def __ApplyTrainingRoundSingle(self):
        self.__generator.GenerateEpisodes(BATCH_SIZE)
        self.__optimizer.zero_grad()
        inputTensor, decisionTensor, destinationValue = self.__generator.GetTrainingTensors()
        logits, value = self.__net(inputTensor)
        lossValue = F.mse_loss(value, destinationValue)
        
        # print("Loss Value " + str(lossValue))
        
        
        advantage = destinationValue - value.detach()
        
        probs = F.softmax(logits, dim = 1)
        
        playedProbs = probs[range(inputTensor.size()[0]), decisionTensor]
        logProbs = F.log_softmax(logits, dim = 1)
        rValue = playedProbs / playedProbs.detach()
       #  print("R -value " + str(rValue))
        
        innerValue = torch.min(advantage * rValue, advantage * torch.clamp(rValue, 0.8, 1.2))
        
        
        lossPolicy = -innerValue.mean()
        #print("Loss Policy " + str(lossPolicy))
        entropy = torch.sum(logProbs * probs, dim = 1)
        lossEntropy =  - entropy.mean()
        #print("Loss Entropy " + str(lossEntropy))
        totalLoss = lossValue + FACTOR_POLICY * lossPolicy + FACTOR_ENTROPY * lossEntropy
        totalLoss.backward()           
        self.__optimizer.step()
            
            
        
    def __TestNet(self):
        runs = 1000
        simulator = simulation.Simulator()
        summe = 0.0
        for _ in range(runs):
            simulator.Reset()
            stop = False
            while not stop:
                state = simulator.GetState()
                goOn  = self.__net.DecideForAction(state)
                stop, score = simulator.ApplyDecision(goOn)
            summe += score
            
        score = summe / runs    
        print("Test Average Score: " + str(score))
        return score 

    
    def TrainNetwork(self):
      
        while True:
            for _ in range(10):
                self.__ApplyTrainingRoundSingle()
            score = self.__TestNet()
            if score > 2.15:
                 torch.save(self.__net, 'NewZombiePPO.dat')
                 break
       
            
            
            
        
   

if __name__ == "__main__":
   trainer = TrainerPPO()
   trainer.TrainNetwork()  
        