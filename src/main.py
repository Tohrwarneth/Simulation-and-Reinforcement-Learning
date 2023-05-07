import sys

from src.logic.Elevator import Elevator
from src.ui.GuiElevator import GuiElevator
from ui.GuiManager import GuiManager
from logic.LogicManager import LogicManager


class Simulation:
    guiActivated: bool = True
    guiManager: GuiManager
    logicManager: LogicManager

    def __int__(self):
        self.init()

    def init(self):
        gui = self.guiActivated
        if gui:
            self.guiManager = GuiManager()

        self.logicManager = LogicManager()

        for i in range(0, 3):
            guiElement = None
            if gui:
                guiElement = GuiElevator(i)
                self.guiManager.addGuiObject(guiElement)
            elevator: Elevator = Elevator(gui=guiElement)
            self.logicManager.addObject(elevator)

        if gui:
            self.guiManager.initObjects()

    def run(self):
        running: bool = True

        while running:
            if self.guiActivated:
                running = self.guiManager.frame()

        self.shutdown()

    def shutdown(self):
        self.guiManager.shutdown()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    simulation = Simulation()
    simulation.init()

    for param in sys.argv:
        if param.__contains__("ui="):
            value = param[3:]
            simulation.guiActivated = value != "false" and value != "False"

    simulation.run()
