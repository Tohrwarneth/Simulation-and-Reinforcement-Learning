from .ILogicObject import ILogicObject
from .Elevator import Elevator
from src.ui.GuiElevator import GuiElevator

class LogicManager:
    logicObjects: list[ILogicObject]

    def __init__(self):
        self.logicObjects = list()

    def addObject(self, object: ILogicObject):
        self.logicObjects.append(object)