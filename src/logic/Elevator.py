from ..ui.GuiElevator import GuiElevator


class Elevator:

    index: int
    capacity: int
    currentFloor: int
    targetFloor: int
    gui: GuiElevator

    def __int__(self, capacity=5, currentFloor=0, gui=None):
        self.capacity = capacity
        self.currentFloor = currentFloor
        self.targetFloor = currentFloor
        self.gui = gui

    def setCurrentFloor(self, floor:int):
        self.currentFloor = floor
        if(self.gui!=None):
            self.gui.setCurrentFloor(floor)