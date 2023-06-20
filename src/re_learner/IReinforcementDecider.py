import enums
from enums import Direction, ElevatorState
from logic.decider_interface import IDecider
from logic.person import Person
from re_learner import Net


class IReinforcementDecider(IDecider):

    @classmethod
    def init(cls, net: Net):
        raise NotImplementedError("IReinforcementDecider: Interface function init not implemented")
    @classmethod
    def get_decision(cls, sim) -> tuple[enums.ElevatorState, enums.ElevatorState, enums.ElevatorState]:
        raise NotImplementedError("IReinforcementDecider: Interface function get_decision not implemented")

    @classmethod
    def search_for_call(cls, position: int, direction: Direction, call_up: list[list[Person]],
                        call_down: list[list[Person]]) -> tuple[int, Direction] | tuple[None, Direction]:
        return None, Direction.UP

    @classmethod
    def get_next_job(cls, position: int, direction: Direction, next_state: ElevatorState, passengers: list[Person]) \
            -> tuple[int, Direction] | tuple[None, ElevatorState]:
        return None, ElevatorState.WAIT
