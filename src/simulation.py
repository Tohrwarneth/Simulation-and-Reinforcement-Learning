import numpy as np
import simpy

from src.utils import Conf
from src.logic.elevator import Elevator
from src.logic.person_manager import PersonManager
from src.ui.gui_manager import GuiManager


class Simulation:
    def __init__(self, eleCap=5, eleSpeed=1,
                 eleWaitingTime=1, visualize=True):
        # private var
        self.env = simpy.Environment()
        self.QueUpward = [[] for _ in range(Conf.max_floor)]
        self.QueDownward = [[] for _ in range(Conf.max_floor)]

        self.stateList = []  # Zustand jeden Takt
        self.log = {"avgWaitingTime": 0, "states": []}

        self.person_manager = PersonManager(self.env, self.QueUpward,
                                            self.QueDownward)

        self.elevatorList = [
            Elevator(i, capacity=eleCap, speed=eleSpeed, waitingTime=eleWaitingTime, enviroment=self.env,
                     QueUpward=self.QueUpward, QueDownward=self.QueDownward)
            for i in range(3)]
        if visualize:
            self.ui_manager = GuiManager(self.env, self.elevatorList, self.QueUpward,
                                         self.QueUpward)
            self.env.process(self.ui_manager.draw())

    def getState(self):
        # TODO
        state = []
        return state

    def getData(self, time):
        self.env.run(time)
        data = self.log
        waitingList = []
        for i in range(len(self.elevatorList)):
            waitingList.extend(self.elevatorList[i].waitingList)
        data["avgWaitingTime"] = np.mean(waitingList)
        data["states"] = self.stateList
        return data


if __name__ == "__main__":
    minToSim = 24 * 60

    S = Simulation(visualize=False)
    data = S.getData(minToSim)
    print(data)
