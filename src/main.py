import sys
from time import time

from src.conf import Conf, Log
from src.logic.elevator.Elevator import Elevator
from logic.LogicManager import LogicManager
from src.ui.GuiElevator import GuiElevator
from ui.GuiManager import GuiManager
from logic.glock import Glock


class Simulation:
    show_gui: bool = True
    guiManager: GuiManager
    logicManager: LogicManager

    def __init__(self, show_gui):
        Log.init()
        self.show_gui = show_gui
        if show_gui:
            self.guiManager = GuiManager()

        self.logicManager = LogicManager()

        for i in range(0, 3):
            guiElement = None
            if show_gui:
                guiElement = GuiElevator(i)
                self.guiManager.addGuiObject(guiElement)
            elevator: Elevator = Elevator(gui=guiElement)
            self.logicManager.addElevator(elevator)

        if show_gui:
            self.guiManager.initObjects()

    def run(self):
        delta_time: float = 1
        running: bool = True

        while running:
            t0 = time()
            if self.show_gui:
                running = self.guiManager.frame(delta_time)
            self.logicManager.update()

            t1 = time()
            delta_time = t1 - t0 if self.show_gui else 1
            Glock.add_delta(delta_time)
            Log.log()
            if Glock.end_of_day:
                break

        self.shutdown()

    def shutdown(self):
        if self.show_gui:
            self.guiManager.shutdown()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    showGui: bool = False
    for param in sys.argv:
        if param.__contains__("ui="):
            value = param[3:]
            showGui = value != "false" and value != "False"
        elif param.__contains__("plots="):
            value = param[3:]
            Conf.show_plots = value != "true" and value != "True"
    simulation = Simulation(showGui)

    simulation.run()
