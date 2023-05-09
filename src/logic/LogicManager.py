from .ILogicObject import ILogicObject
from .elevator.Elevator import Elevator
from .person.person_manager import PersonManager


class LogicManager:
    person_manager: PersonManager
    elevators: list[Elevator]

    def __init__(self):
        self.person_manager = PersonManager()

    def update(self):
        self.person_manager.update()

    def addElevator(self, elevator):
        pass