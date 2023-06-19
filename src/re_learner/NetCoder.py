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
    capacity = elevator_capacity
    stateWidth = len(normalize_game_state(game_state))


def encode_in_tensor(tensor, tensor_position, game_state):
    global stateWidth
    n_game_state = normalize_game_state(game_state)
    for index, element in enumerate(n_game_state):
        tensor[tensor_position, index] = element


def normalize_game_state(game_state: tuple) -> list:
    global capacity
    avg_waiting, remaining, call_up, call_down, elevators = game_state

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

    return [avg_waiting, remaining] + n_call_up + n_call_down + n_elevators
