from src.ui.elevator.GuiElevator import GuiElevator
import simpy


class Elevator:
    index: int
    capacity: int
    passengers: simpy.Store
    currentFloor: int
    targetFloor: int
    jobs: list[int]
    person_floor: tuple[int, int]
    gui: GuiElevator

    def __init__(self, index: int, env: simpy.Environment = None, capacity: int = 5, current_floor: int = 0, gui: GuiElevator = None):
        self.index = index
        self.capacity = capacity
        self.currentFloor = current_floor
        self.targetFloor = current_floor
        self.gui = gui
        self.jobs = list()
        self.person_floor = (0, 0)
        self.passengers = simpy.Store(env, capacity)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(index={self.index}, currentFloor={self.currentFloor})"

    def init(self, env: simpy.Environment, capacity=5, current_floor=0, gui=None):
        self.capacity = capacity
        self.currentFloor = current_floor
        self.targetFloor = current_floor
        self.gui = gui

    def update(self, tact: int, person_floor: tuple[int, int]) -> list:
        self.person_floor = person_floor
        if not self.gui == None:
            self.gui.person_floor = person_floor
        job_log = self.jobs
        job_log += [None for i in range(len(self.jobs), self.capacity)]
        return [self.currentFloor, self.targetFloor] + list(self.passengers.items) + self.jobs

    def set_current_floor(self, floor: int):
        self.currentFloor = floor
        if floor == self.targetFloor:
            self.finished_job(floor)
        if not self.gui == None:
            self.gui.setCurrentFloor(floor)

    def add_job(self, floor: int):
        self.jobs.append(floor)
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
