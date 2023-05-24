from logic.decider_interface import IDecider
from enums import Direction, ElevatorState
from logic.person import Person


class ReinforcementDecider(IDecider):
    pass
    # @staticmethod
    # def search_for_call(position: int, direction: Direction, call_up: list[list[Person]],
    #                     call_down: list[list[Person]]) -> tuple[int, Direction] | tuple[None, Direction]:
    #     pass
    #
    # @staticmethod
    # def get_next_job(position: int, direction: Direction, next_state: ElevatorState, passengers: list[Person]) \
    #         -> tuple[int, Direction] | tuple[None, ElevatorState]:
    #     pass
