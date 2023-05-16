import sys
from time import time

import numpy as np
import simpy

from src.utils import Conf, Clock
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

    def run(self):

        t_old: float = time()
        while Clock.running:
            t_new: float = time()
            for _ in range(Clock.tactBuffer):
                self.personManager.manage()
                for elevator in self.elevatorList:
                    elevator.operate()

            if self.showGui:
                self.ui_manager.draw()
                Clock.tact += 1

            Clock.tactBuffer = 0

            delta_time: float = (t_new - t_old) * Clock.speedScale if self.showGui else 1
            Clock.add_time(delta_time)

        self.shutdown()

    def shutdown(self):
        data = S.getData()
        print(data)

    def getState(self):
        # TODO
        state = []
        return state

    def getData(self):
        # TODO: Logs
        data = self.log
        waitingList = []
        for i in range(len(self.elevatorList)):
            waitingList.extend(self.elevatorList[i].waitingList)
        data["avgWaitingTime"] = np.mean(waitingList)
        data["states"] = self.stateList
        return data


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
            Conf.skip = int(value)

    S = Simulation(show_gui=showGui)
