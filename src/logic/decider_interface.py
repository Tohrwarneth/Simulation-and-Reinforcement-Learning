from enums import Direction, ElevatorState
from logic.person import Person


class IDecider:
    @staticmethod
    def search_for_call(position: int, direction: Direction, call_up: list[list[Person]],
                        call_down: list[list[Person]]) -> tuple[int, Direction] | tuple[None, Direction]:
        raise NotImplementedError("IDecider: Interface function search_for_call not implemented")

    @staticmethod
    def get_next_job(position: int, direction: Direction, next_state: ElevatorState, passengers: list[Person]) \
            -> tuple[int, Direction] | tuple[None, ElevatorState]:
        raise NotImplementedError("IDecider: Interface function of get_next_job not implemented")
