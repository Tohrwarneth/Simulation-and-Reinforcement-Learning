import sys
from time import time

from src.conf import Conf, Log
from src.logic.elevator.Elevator import Elevator
from logic.LogicManager import LogicManager
from src.ui.GuiFloor import GuiFloor
from src.ui.elevator.GuiElevator import GuiElevator
from ui.GuiManager import GuiManager
from src.glock import Glock


class Simulation:
    show_gui: bool = True
    step_gui: bool = False
    skipped: bool
    guiManager: GuiManager
    logicManager: LogicManager

    def __init__(self, show_gui: bool, step_gui: bool):
        Log.init()
        self.skipped = Conf.skip == 0
        self.show_gui = show_gui
        self.step_gui = step_gui
        gui_element = None
        if show_gui:
            self.guiManager = GuiManager(step_gui)
            gui_element = GuiFloor()
            self.guiManager.add_gui_object(gui_element)

        self.logicManager = LogicManager(gui_floor=gui_element)

        for i in range(0, 3):
            if show_gui:
                gui_element = GuiElevator(i)
                self.guiManager.add_gui_object(gui_element)
            elevator: Elevator = Elevator(gui=gui_element)
            self.logicManager.addElevator(elevator)

        if show_gui:
            self.guiManager.initObjects()

    def run(self):
        delta_time: float = 1
        running: bool = True
        if not self.skipped:
            Conf.speed_scale = 256

        while running:
            t0 = time()
            h, _ = Glock.get_time()
            if not self.skipped and h == Conf.skip:
                Conf.speed_scale = 1
                self.skipped = True
            self.logicManager.update()

            if self.show_gui:
                running = self.guiManager.frame(delta_time)

            t1 = time()
            delta_time = (t1 - t0) * Conf.speed_scale if self.show_gui else 1
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
    step_gui: bool = False
    for param in sys.argv:
        if param.__contains__("ui="):
            value = param[3:]
            showGui = value != "false" and value != "False"
        elif param.__contains__("plots="):
            value = param[6:]
            Conf.show_plots = value != "true" and value != "True"
        elif param.__contains__("step="):
            value = param[5:]
            step_gui = value != "true" and value != "True"
        elif param.__contains__("skip="):
            value = param[5:]
            Conf.skip = int(value)

    if not showGui and step_gui:
        step_gui = False
    simulation = Simulation(showGui, step_gui)

    simulation.run()
