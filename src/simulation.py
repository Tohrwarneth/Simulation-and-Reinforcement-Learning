import sys
from time import time
import numpy as np
from src.utils import Conf, Clock, Logger
from src.logic.elevator import Elevator
from src.logic.person_manager import PersonManager
from src.ui.gui_manager import GuiManager


class Simulation:
    def __init__(self, eleCap=5, eleSpeed=1,
                 eleWaitingTime=1, show_gui=True):
        # private var
        self.showGui = show_gui
        self.callUp = [[] for _ in range(Conf.maxFloor)]
        self.callDown = [[] for _ in range(Conf.maxFloor)]

        self.stateList = []  # Zustand jeden Takt
        self.log = {"avgWaitingTime": 0, "states": []}

        self.personManager = PersonManager(self.callUp,
                                           self.callDown)

        self.elevatorList = [
            Elevator(capacity=eleCap, speed=eleSpeed, waitingTime=eleWaitingTime,
                     QueUpward=self.callUp, QueDownward=self.callDown)
            for _ in range(3)]
        if show_gui:
            self.ui_manager = GuiManager(self.elevatorList, self.callUp,
                                         self.callUp)
        self.run()

    def run(self):

        Logger.init()

        if Clock.skip:
            Clock.speedScale = 265

        t_old: float = time()
        while Clock.running:
            t_new: float = time()
            for _ in range(Clock.tactBuffer):
                if Clock.skip and Clock.tact / 60 >= Clock.skip:
                    Clock.skip = None
                    Clock.speedScale = 1.0
                Logger.new_tact()
                self.personManager.manage()
                for elevator in self.elevatorList:
                    elevator.operate()
                Clock.tact += 1
                Logger.log()

            if self.showGui:
                self.ui_manager.draw()

            Clock.tactBuffer = 0

            delta_time: float = (t_new - t_old) * Clock.speedScale if self.showGui else 1
            t_old = t_new
            if not Clock.pause:
                Clock.add_time(delta_time)

        self.shutdown()

    def shutdown(self):
        Clock.end_of_day = True
        data = self.end_of_day_log()
        print(data)

    def end_of_day_log(self):
        # TODO: EOD Log verbessern für Reinforcement und einmal für Log Ordner
        Logger.new_tact()
        log = self.personManager.end_of_day()
        for elevator in self.elevatorList:
            log = log | elevator.end_of_day()

        waitingList = []
        for i in range(len(self.elevatorList)):
            waitingList.extend(self.elevatorList[i].waitingList)
        log["avgWaitingTime"] = np.mean(waitingList)

        Logger.add_data(log)
        Logger.log()
        return log


if __name__ == "__main__":
    showGui: bool = False
    step_gui: bool = False
    for param in sys.argv:
        if param.__contains__("ui="):
            value = param[3:]
            showGui = value != "false" and value != "False"
        elif param.__contains__("plots="):
            value = param[6:]
            Conf.showPlots = value == "true" or value == "True"
        elif param.__contains__("step="):
            value = param[5:]
            step_gui = value != "true" and value != "True"
        elif param.__contains__("skip="):
            value = param[5:]
            Clock.skip = int(value)

    S = Simulation(show_gui=showGui)
