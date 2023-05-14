from src.logic.states import ElevatorState
from src.logic.person.Person import Person
from src.ui.elevator.GuiElevator import GuiElevator
import simpy


class Elevator:
    index: int
    capacity: int
    state: ElevatorState
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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(index={self.index}, currentFloor={self.currentFloor})"

    def update(self, tact: int, person_floor: tuple[list[Person], list[Person]]) -> list:
        (up, down) = (len(person_floor[0]), len(person_floor[1]))
        self.person_floor = (up, down)
        if not self.gui == None:
            self.gui.person_floor = self.person_floor
            self.gui.state = self.state
        job_log = self.jobs
        job_log += [None for i in range(len(self.jobs), self.capacity)]

        if self.state == ElevatorState.WAIT:
            up, down = self.person_floor
            if up > 0 or down > 0:
                pass
                 # TODO: Frag nach leuten an. In eine extra Methode. Und da dann so viele, wie Kapazitäten frei sind.
        else:
            if self.currentFloor == self.targetFloor:
                self.manager.reached_floor([person for person in self.passengers
                                            if person.targetFloor == self.targetFloor])
            else:
                self.currentFloor += 1 if self.state == ElevatorState.UP else -1

        return [self.currentFloor, self.targetFloor] + self.passengers + self.jobs

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
        job_list: list[str] = [f"({self.index}) {i}. job" for i in range(0, self.capacity)]
        return [f"({self.index}) current floor", f"({self.index}) target floor",
                f"({self.index}) passengers"] + job_list

    def get_next_floor(self) -> int | None:
        for job in self.jobs:
            if self.state == ElevatorState.UP and job > self.currentFloor:
                return job
            elif self.state == ElevatorState.DOWN and job < self.currentFloor:
                return job
        return self.currentFloor
