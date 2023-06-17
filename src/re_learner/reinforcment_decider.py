from logic.decider_interface import IDecider
from enums import Direction, ElevatorState
from logic.person import Person
from re_learner import Net


class ReinforcementDecider(IDecider):
    simulation = None
    net: Net

    @classmethod
    def init(cls, simulation, net: Net):
        cls.net = net
        cls.simulation = simulation

    @classmethod
    def search_for_call(cls, position: int, direction: Direction, call_up: list[list[Person]],
                        call_down: list[list[Person]]) -> tuple[int, Direction] | tuple[None, Direction]:
        return None, Direction.UP

    @classmethod
    def get_next_job(cls, position: int, direction: Direction, next_state: ElevatorState, passengers: list[Person]) \
            -> tuple[int, Direction] | tuple[None, ElevatorState]:
        return None, ElevatorState.WAIT
