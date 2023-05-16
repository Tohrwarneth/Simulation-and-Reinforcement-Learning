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
    DOWN = 2
    WAIT = 3

    def __eq__(self, other):
        if isinstance(other, ElevatorState):
            return self.value == other.value and self.name == other.name
