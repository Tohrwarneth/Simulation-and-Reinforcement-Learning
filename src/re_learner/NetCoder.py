from __future__ import annotations
"""
Class created by Prof. C. Lürig
modified from Zombie Dice to Elevator Simulation
"""

import utils
import enums

stateWidth = 0
capacity = 0
decisions_states = dict()
num_actions: int


def init(game_state, elevator_capacity):
    '''
    Transfer all possible decisions of triples to indices and fills look-up dictionary

    :param game_state: example game state
    :param elevator_capacity: number elevators
    :return:
    '''
    global stateWidth, capacity, num_actions
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
    num_actions = len(decisions_states)


def encode_in_tensor(tensor, tensor_position, game_state) -> None:
    '''
    Insert game state into tensor

    :param tensor: target tensor
    :param tensor_position: which position within tensor
    :param game_state: current game state
    '''
    global stateWidth
    n_game_state = normalize_game_state(game_state)
    for index, element in enumerate(n_game_state):
        tensor[tensor_position, index] = element


def decision_to_states(decisions: int) -> \
        tuple[enums.ElevatorState, enums.ElevatorState, enums.ElevatorState] | tuple[None, None, None]:
    '''
    Translate index to decision tuple

    :param decisions: decision index
    :return: decision tuple
    '''
    global decisions_states
    if decisions is None:
        return None, None, None
    assert isinstance(decisions, int)
    decisions = decisions % (3 * 3 * 3)
    states = decisions_states[decisions]
    return states


def states_to_decision(states: tuple[enums.ElevatorState, enums.ElevatorState, enums.ElevatorState]) -> int:
    '''
    Returns index of decision tuple

    :param states: tuple of decisions
    :return: decision index
    '''
    global decisions_states
    key = [k for k, v in decisions_states.items() if v == states][0]
    return key


def normalize_game_state(game_state: tuple) -> list:
    '''
    Encode current game state to tensor friendly data types

    :param game_state: current game state
    :return: tensor friendly list
    '''
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
            n_elevators.append(-1)
    return [tact / 60 * 24, avg_waiting / 60 * 24, remaining / conf.totalAmountPerson] \
        + n_call_up + n_call_down + n_elevators
