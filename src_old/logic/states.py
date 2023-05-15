from enum import Enum


class DayState(Enum):
    MORNING = 1
    PRE_LUNCH = 2
    POST_LUNCH = 3
    EVENING = 4
    AT_HOME = 5

    def __eq__(self, other):
        return self.value == other.value


class Direction(Enum):
    UP = 0
    DOWN = 1

    def __eq__(self, other):
        return self.value == other.value


class ElevatorState(Enum):
    WAIT = 1
    UP = 2
    DOWN = 3

    def __eq__(self, other):
        return self.value == other.value
