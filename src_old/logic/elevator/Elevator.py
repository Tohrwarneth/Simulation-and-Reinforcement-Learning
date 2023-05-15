from src_old.conf import Conf
from src_old.logic.states import ElevatorState, Direction
from src_old.logic.person.Person import Person
from src_old.ui.elevator.GuiElevator import GuiElevator
import simpy


class Elevator:
    index: int
    capacity: int
    state: ElevatorState
    direction_state: ElevatorState
    passengers: list[Person]
    currentFloor: int
    targetFloor: int
    jobs: list[int]
    person_floor: tuple[int, int]
    gui: GuiElevator
    manager = None  # Logic Manager

    def __init__(self, manager, index: int, env: simpy.Environment = None, capacity: int = 5,
                 current_floor: int = 0,
                 gui: GuiElevator = None):
        self.manager = manager
        self.index = index
        self.capacity = capacity
        self.currentFloor = current_floor
        self.targetFloor = current_floor
        self.gui = gui
        self.jobs = list()
        self.person_floor = (0, 0)
        self.passengers = list()
        self.state = ElevatorState.WAIT
        self.direction_state = ElevatorState.WAIT

    def __repr__(self) -> str:
        return f"{type(self).__name__}(index={self.index}, currentFloor={self.currentFloor})"

    def update(self, person_floor: tuple[list[Person], list[Person]]) -> list:
        (up, down) = (len(person_floor[Direction.UP.value]), len(person_floor[Direction.DOWN.value]))
        self.person_floor = (up, down)
        if not self.gui == None:
            self.gui.person_floor = self.person_floor
            self.gui.state = self.state
            self.gui.passengers = len(self.passengers)
        job_log = self.jobs.copy()
        job_log += [None for i in range(len(self.jobs), self.capacity)]

        if self.state == ElevatorState.WAIT:
            up, down = self.person_floor
            if (up > 0 or down > 0) and len(self.passengers) < self.capacity:
                persons = self.manager.load_elevator(self, Direction.UP, self.capacity - len(self.passengers))
                self.passengers += persons
                num_pos, num_neg = 0, 0
                for p in persons:
                    self.add_job(p.targetFloor)
                    if p.targetFloor > self.currentFloor:
                        num_pos += 1
                    else:
                        num_neg += 1
                if num_pos > num_neg and self.state == ElevatorState.WAIT:
                    self.state = ElevatorState.UP
                    self.direction_state = ElevatorState.UP
                    self.targetFloor = min([i for i in self.jobs if i > self.currentFloor])
                else:
                    self.state = ElevatorState.DOWN
                    self.direction_state = ElevatorState.DOWN
                    self.targetFloor = max([i for i in self.jobs if i < self.currentFloor])
            elif not self.targetFloor == self.currentFloor:
                if self.targetFloor < self.currentFloor:
                    self.direction_state = ElevatorState.DOWN
                elif self.targetFloor > self.currentFloor:
                    self.direction_state = ElevatorState.UP

        else:
            if self.currentFloor == self.targetFloor:
                self.state = ElevatorState.WAIT
                leaver = [person for person in self.passengers
                          if person.targetFloor == self.targetFloor]
                for person in leaver:
                    self.passengers.remove(person)
                    self.jobs.remove(person.targetFloor)
                self.manager.reached_floor(leaver)
                if len(self.jobs) == 0:
                    self.direction_state = ElevatorState.WAIT
            else:
                self.currentFloor += 1 if self.state == ElevatorState.UP else -1
                if self.currentFloor == Conf.max_floor - 1:
                    self.direction_state = ElevatorState.DOWN
                elif self.currentFloor == 0:
                    self.direction_state = ElevatorState.UP
                elif len(self.jobs) == 0:
                    self.direction_state = ElevatorState.WAIT

        log_job = "("
        for j, job in enumerate(self.jobs):
            log_job += f"{job},"
        log_job += ")"
        return [self.currentFloor, self.targetFloor, log_job, self.state.name, self.direction_state.name]

    def enter(self, person: Person):
        self.passengers.append(person)
        self.add_job(person.currentFloor)

    def leave(self, person: Person):
        self.passengers.remove(person)
        self.finished_job(person.currentFloor)

    def set_current_floor(self, floor: int):
        self.currentFloor = floor
        if floor == self.targetFloor:
            self.finished_job(floor)
        if not self.gui == None:
            self.gui.setCurrentFloor(floor)

    def add_job(self, floor: int):
        self.jobs.append(floor)
        self.jobs.sort()
        if not self.gui == None:
            self.gui.setJobs(self.jobs)

    def finished_job(self, floor: int):
        self.jobs.remove(floor)
        if not self.gui == None:
            self.gui.setJobs(self.jobs)

    def get_log_header(self) -> list[str]:
        return [f"({self.index}) current floor", f"({self.index}) target floor", f"({self.index}) job",
                f"({self.index}) state", f"({self.index}) direction state"]

    def get_next_floor(self) -> int | None:
        for job in self.jobs:
            if self.state == ElevatorState.UP and job > self.currentFloor:
                return job
            elif self.state == ElevatorState.DOWN and job < self.currentFloor:
                return job
        return self.currentFloor
