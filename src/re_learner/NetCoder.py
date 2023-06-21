from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Created on Sun May 23 13:02:21 2021

@author: chris
"""
'''Contains helper functions to encode the state of the game for neuronal networks.'''
import utils
import enums

stateWidth = 0
capacity = 0
decisions_states = dict()


def init(game_state, elevator_capacity):
    global stateWidth, capacity
    capacity = elevator_capacity
    stateWidth = len(normalize_game_state(game_state))

    index = 0
    for elev_1 in range(3):
        for elev_2 in range(3):
            for elev_3 in range(3):
                decisions_states[index] = (enums.ElevatorState.get_value_by_index(elev_1),
                                           enums.ElevatorState.get_value_by_index(elev_2),
                                           enums.ElevatorState.get_value_by_index(elev_3))
                index += 1


def encode_in_tensor(tensor, tensor_position, game_state):
    global stateWidth
    n_game_state = normalize_game_state(game_state)
    for index, element in enumerate(n_game_state):
        tensor[tensor_position, index] = element


def decision_to_states(decisions: int) -> \
        tuple[enums.ElevatorState, enums.ElevatorState, enums.ElevatorState]:
    global decisions_states
    assert isinstance(decisions, int)
    decisions = decisions % (3 * 3 * 3)
    states = decisions_states[decisions]
    return states


def states_to_decision(states: tuple[enums.ElevatorState, enums.ElevatorState, enums.ElevatorState]) -> int:
    global decisions_states
    key = [k for k, v in decisions_states.items() if v == states][0]
    return key


def normalize_game_state(game_state: tuple) -> list:
    # TODO: Am schluss normieren
    conf = utils.Conf
    global capacity
    tact, avg_waiting, remaining, call_up, call_down, elevators = game_state

    n_call_up = [1 if len(persons) > 0 else 0 for persons in call_up]
    n_call_down = [1 if len(persons) > 0 else 0 for persons in call_down]

    n_elevators = list()
    for elevator in elevators:
        position, direction, next_state, passengers = elevator
        n_elevators.append(position / conf.maxFloor)
        n_elevators.append(direction.value)
        n_elevators.append(next_state.value / 3)
        empty_job = capacity - len(passengers)
        for person in passengers:
            n_elevators.append(person.schedule[0][1] / conf.maxFloor)  # target floor
        for i in range(empty_job):
            # TODO: passt die -1?
            n_elevators.append(-1)

    # avg_waiting / 60 * 24 = 1, wenn jemand bei Takt 0 kommt und dann bis Tagende wartet
    # Wird nie 1
    return [tact / 60 * 24, avg_waiting / 60 * 24, remaining / conf.totalAmountPerson] \
        + n_call_up + n_call_down + n_elevators
