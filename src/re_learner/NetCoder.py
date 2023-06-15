# -*- coding: utf-8 -*-
"""
Created on Sun May 23 13:02:21 2021

@author: chris
"""

'''Contains helper functions to encode the state of the game for neuronal networks.'''

StateWidth = 6

def SetStateWidth(gameState):
    elevators, call_up, call_down = gameState
    StateWidth = 0

    for e in elevators:
        StateWidth += len(e)


def EncodeInTensor(tensor, tensor_position, gameState):
    elevators, call_up, call_down = gameState
    # TODO: Nur int und floats im tensor

    # Elevator
    for index, elevator in enumerate(elevators):
        position, direction, next_state, passengers = elevator
        tensor_index = index * (len(elevator) + len(passengers) - 1)
        tensor[tensor_position, 0 + tensor_index] = position
        tensor[tensor_position, 1 + tensor_index] = direction
        tensor[tensor_position, 2 + tensor_index] = next_state
        for j, person in enumerate(passengers):
            tensor[tensor_position, 3 + tensor_index + j] = person.schedule[0][1]

    # Office
    tensor[tensor_position, 4] = call_up
    tensor[tensor_position, 5] = call_down


def EncodeInTensorPlain(tensor, position, gameState):
    # in sack
    tensor[position, 0] = gameState[0]['red']
    tensor[position, 1] = gameState[0]['yellow']
    tensor[position, 2] = gameState[0]['green']
    # on table
    tensor[position, 3] = gameState[1]['red']
    tensor[position, 4] = gameState[1]['yellow']
    tensor[position, 5] = gameState[1]['green']
    # shots
    tensor[position, 6] = gameState[2]
    # brains
    tensor[position, 7] = gameState[3]
