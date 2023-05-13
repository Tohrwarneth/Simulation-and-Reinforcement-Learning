import sys
from time import time

import simpy

from src.conf import Conf, Log, LogData
from src.logic.elevator.Elevator import Elevator
from logic.LogicManager import LogicManager
from src.ui.GuiFloor import GuiFloor
from src.ui.elevator.GuiElevator import GuiElevator
from ui.GuiManager import GuiManager
from src.clock import Clock


class Simulation:
    show_gui: bool = True
    step_gui: bool = False
    skipped: bool
    guiManager: GuiManager
    logicManager: LogicManager
    env: simpy.Environment

    def __init__(self, show_gui: bool, step_gui: bool):
        self.skipped = Conf.skip == 0
        self.show_gui = show_gui
        self.step_gui = step_gui
        self.env = simpy.Environment()
        gui_element = None
        if show_gui:
            self.guiManager = GuiManager(step_gui)
            gui_element = GuiFloor()
            self.guiManager.add_gui_object(gui_element)

        self.logicManager = LogicManager(self.env, gui=gui_element)

        for i in range(0, 3):
            if show_gui:
                gui_element = GuiElevator(i)
                self.guiManager.add_gui_object(gui_element)
            elevator: Elevator = Elevator(i, env=self.env, gui=gui_element)
            self.logicManager.add_elevator(elevator)

        if show_gui:
            self.guiManager.initObjects()

        LogData.add_header(self.logicManager.get_log_header())
        Log.init()

    def run(self):
        delta_time: float = 0
        old_tact: int = 0
        running: bool = True
        self.logicManager.update(0)
        if not self.skipped:
            Conf.speed_scale = 256

        t_old = time()
        while running:
            t_new = time()
            h, s = Clock.get_time()
            if not self.skipped and h == Conf.skip:
                Conf.speed_scale = 1
                self.skipped = True

            for tact in range(old_tact, Clock.tact):
                log_data: LogData = self.logicManager.update(tact)
                Log.log(log_data)

            if self.show_gui:
                running = self.guiManager.frame(delta_time)

            delta_time = (t_new - t_old) * Conf.speed_scale if self.show_gui else 1
            t_old = t_new
            old_tact = Clock.tact
            Clock.add_delta(delta_time)
            if Clock.end_of_day:
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
            Conf.show_plots = value == "true" or value == "True"
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
