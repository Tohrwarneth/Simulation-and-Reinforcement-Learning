import sys
from time import time

import simpy

from src_old.conf import Conf, Log, LogData
from src_old.logic.elevator.Elevator import Elevator
from logic.LogicManager import LogicManager
from src_old.ui.GuiFloor import GuiFloor
from src_old.ui.elevator.GuiElevator import GuiElevator
from ui.GuiManager import GuiManager
from src_old.clock import Clock


class Simulation:
    show_gui: bool = True
    step_gui: bool = False
    skipped: bool
    running: bool = True
    guiManager: GuiManager
    logicManager: LogicManager
    env: simpy.Environment

    def __init__(self, show_gui: bool, step_gui: bool):
        self.running = True
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
            elevator: Elevator = Elevator(self.logicManager, i, env=self.env, gui=gui_element)
            self.logicManager.add_elevator(elevator)

        if show_gui:
            self.guiManager.initObjects()

        LogData.add_header(self.logicManager.get_log_header())
        Log.init()
        Clock.env = self.env
        self.env.process(self.run())

    def run(self):
        old_tact: int = 0
        if not self.skipped:
            Conf.speed_scale = 256

        t_old = time()
        while self.running:
            t_new = time()
            h, m = Clock.get_time()
            if not self.skipped and h == Conf.skip:
                Conf.speed_scale = 1
                self.skipped = True

            log = self.logicManager.update()
            if not old_tact == int(self.env.now):
                Log.log(log)
            if self.show_gui:
                self.running = self.guiManager.frame()

            # delta_time = (t_new - t_old) * Conf.speed_scale if self.show_gui else 1
            self.delta_time = 0.01 * Conf.speed_scale if self.show_gui else 1
            t_old = t_new
            Clock.add_delta(self.delta_time)
            if Clock.end_of_day:
                break
            old_tact = Clock.tact
            yield self.env.timeout(Clock.delta_time)

        self.shutdown()

    def draw_gui_frame(self):
        while self.running:
            self.running = self.guiManager.frame()
            yield self.env.timeout(Clock.delta_time)

    def shutdown(self):
        self.logicManager.eod()
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
    simulation.env.run(until=24 * 60)
    simulation.shutdown()
