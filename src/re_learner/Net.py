# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 07:51:46 2020

@author: chris
"""

import torch
import torch.nn as nn
import NetCoder
import random

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
       
        activation = nn.LeakyReLU()
        self.__common = nn.Sequential(nn.Linear(8, 20), activation, 
                                      nn.Linear(20, 40), activation,
                                      nn.Linear(40, 20), activation)
        
        # We know that value >= 0
        self.__value = nn.Sequential( 
                                    nn.Linear(20, 10), activation, 
                                      nn.Linear(10, 1), activation)
        
        self.__policy = nn.Sequential(
                                    nn.Linear(20, 10), activation, 
                                      nn.Linear(10, 2))
        
      
        
    def forward(self, x):
        base = self.__common(x)
        value = self.__value(base) * 13.0
        # These will be all 1 element arrays.
        value = value.squeeze(-1)
        logits = self.__policy(base)
        # Policy is returned in logits
        return logits, value
    
        
        
    def DecideForAction(self, gameState):
        inTensor = torch.zeros((1, NetCoder.StateWidth), dtype=torch.float32)
        NetCoder.EncodeInTensor(inTensor, 0, gameState)
        with torch.no_grad():
            logits, value = self(inTensor)
            probs = torch.nn.functional.softmax(logits, dim = 1)
            probs = probs[0].data.cpu().numpy()
        
            
        return random.uniform(0.0, 1.0) < probs[1]
            
    

        