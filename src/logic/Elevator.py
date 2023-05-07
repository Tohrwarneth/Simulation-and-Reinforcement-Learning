from src.ui.GuiElevator import GuiElevator
from .ILogicObject import ILogicObject


class Elevator(ILogicObject):
    index: int
    capacity: int
    currentFloor: int
    targetFloor: int
    gui: GuiElevator

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, capacity=5, currentFloor=0, gui=None):
        self.capacity = capacity
        self.currentFloor = currentFloor
        self.targetFloor = currentFloor
        self.gui = gui

    def __repr__(self) -> str:
        return f"{type(self).__name__}(index={self.index}, currentFloor={self.currentFloor})"

    def init(self, capacity=5, currentFloor=0, gui=None):
        self.capacity = capacity
        self.currentFloor = currentFloor
        self.targetFloor = currentFloor
        self.gui = gui

    def update(self):
        pass

    def setCurrentFloor(self, floor: int):
        self.currentFloor = floor
        if self.gui != None:
            self.gui.setCurrentFloor(floor)
