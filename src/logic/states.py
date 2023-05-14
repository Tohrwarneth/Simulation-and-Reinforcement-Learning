from enum import Enum


class DayState(Enum):
    MORNING = 1
    PRE_LUNCH = 2
    POST_LUNCH = 3
    EVENING = 4
    AT_HOME = 5


class ElevatorState(Enum):
    WAIT = 1
    UP = 2
    DOWN = 3
