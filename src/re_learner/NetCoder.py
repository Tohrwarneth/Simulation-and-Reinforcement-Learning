# -*- coding: utf-8 -*-
"""
Created on Sun May 23 13:02:21 2021

@author: chris
"""
'''Contains helper functions to encode the state of the game for neuronal networks.'''

stateWidth = 0
capacity = 0


def init(game_state, elevator_capacity):
    global stateWidth, capacity
    stateWidth = len(normalize_game_state(game_state))
    capacity = elevator_capacity


def encode_in_tensor(tensor, tensor_position, game_state):
    n_game_state = normalize_game_state(game_state)
    for index, element in enumerate(n_game_state):
        tensor[tensor_position, index] = element

#
# def EncodeInTensorPlain(tensor, position, gameState):
#     # in sack
#     tensor[position, 0] = gameState[0]['red']
#     tensor[position, 1] = gameState[0]['yellow']
#     tensor[position, 2] = gameState[0]['green']
#     # on table
#     tensor[position, 3] = gameState[1]['red']
#     tensor[position, 4] = gameState[1]['yellow']
#     tensor[position, 5] = gameState[1]['green']
#     # shots
#     tensor[position, 6] = gameState[2]
#     # brains
#     tensor[position, 7] = gameState[3]


def normalize_game_state(game_state: tuple) -> list:
    global capacity
    call_up, call_down, elevators = game_state

    n_call_up = [1 if len(persons) > 0 else 0 for persons in call_up]
    n_call_down = [1 if len(persons) > 0 else 0 for persons in call_down]

    n_elevators = list()
    for elevator in elevators:
        position, direction, next_state, passengers = elevator
        n_elevators.append(position)
        n_elevators.append(direction.value)
        n_elevators.append(next_state.value)
        empty_job = capacity - len(passengers)
        for person in passengers:
            n_elevators.append(person.schedule[0][1])  # target floor
        for i in range(empty_job):
            n_elevators.append(-1)

    return n_call_up + n_call_down + n_elevators
