from enum import Enum


class Direction(Enum):
    UP = 0
    DOWN = 1

    def __eq__(self, other):
        if isinstance(other, Direction):
            return self.value == other.value and self.name == other.name
        elif isinstance(other, int):
            return self.value == other


class ElevatorState(Enum):
    UP = 1
    WAIT = 0
    DOWN = -1

    @staticmethod
    def get_value_by_index(index: int):
        if index == 1:
            return ElevatorState.UP
        elif index == 0:
            return ElevatorState.WAIT
        elif index == -1:
            return ElevatorState.DOWN

    def __eq__(self, other):
        if isinstance(other, ElevatorState):
            return self.value == other.value and self.name == other.name
