from src.ui.elevator.GuiElevator import GuiElevator


class Elevator:
    index: int
    capacity: int
    currentFloor: int
    targetFloor: int
    jobs: list[int]
    person_floor: tuple[int, int]
    gui: GuiElevator

    def __init__(self, capacity=5, current_floor=0, gui=None):
        self.capacity = capacity
        self.currentFloor = current_floor
        self.targetFloor = current_floor
        self.gui = gui
        self.jobs = list()
        self.person_floor = list()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(index={self.index}, currentFloor={self.currentFloor})"

    def init(self, capacity=5, currentFloor=0, gui=None):
        self.capacity = capacity
        self.currentFloor = currentFloor
        self.targetFloor = currentFloor
        self.gui = gui

    def update(self, person_floor):
        self.person_floor = person_floor
        if not self.gui == None:
            self.gui.person_floor = person_floor

    def setCurrentFloor(self, floor: int):
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
