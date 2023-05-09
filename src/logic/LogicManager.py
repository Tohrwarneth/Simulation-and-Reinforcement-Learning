from .elevator.Elevator import Elevator
from .person.person_manager import PersonManager
from src.ui.GuiFloor import GuiFloor


class LogicManager:
    person_manager: PersonManager
    elevators: list[Elevator]

    def __init__(self, gui_floor: GuiFloor= None):
        self.person_manager = PersonManager()
        self.person_manager.init(gui = gui_floor)
        self.elevators = list()
        self.person_floor = list()

    def update(self):
        self.person_manager.update()
        for elevator in self.elevators:
            elevator.update(self.person_manager.person_floor[elevator.currentFloor])

    def addElevator(self, elevator):
        self.elevators.append(elevator)
