from ..ILogicObject import ILogicObject
from ..states import DayState


class Person:
    index: int
    currentFloor: int
    targetFloor: int
    currentJob: tuple[int, int]
    schedule: list[(int, int)]

    def __init__(self, index: int, schedule: list[tuple[int, int]]):
        self.index = index
        self.currentFloor = 0
        self.schedule = schedule
        job = schedule[0]
        self.targetFloor = job[1]
        self.currentJob = None

    def get_job(self) -> tuple[int, int]:
        job = self.schedule[0]
        self.targetFloor = job[1]
        return job

    def next_job(self) -> tuple[int, int]:
        self.schedule.pop(0)
        job = self.schedule[0]
        self.targetFloor = job[1]
        return job

    def __repr__(self):
        return f"{self.index},{self.targetFloor},{self.schedule}"
