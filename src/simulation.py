import sys
from time import time
import numpy as np
from matplotlib import pyplot as plt

from src.utils import Conf, Clock, Logger
from src.logic.elevator import Elevator
from src.logic.person_manager import PersonManager
from src.ui.gui_manager import GuiManager


class Simulation:
    avgWaitingTime: list[float]

    def __init__(self, eleCap=5, eleSpeed=1,
                 eleWaitingTime=1, show_gui=True):
        # private var
        self.showGui = show_gui
        self.avgWaitingTime = list()
        self.callUp = [[] for _ in range(Conf.maxFloor)]
        self.callDown = [[] for _ in range(Conf.maxFloor)]

        self.stateList = []  # Zustand jeden Takt
        self.log = {"avgWaitingTime": 0, "states": []}

        self.personManager = PersonManager(self.callUp,
                                           self.callDown)

        self.elevatorList = list()
        for _ in range(3):
            elevator = Elevator(capacity=eleCap, speed=eleSpeed, waitingTime=eleWaitingTime,
                                call_up=self.callUp, call_down=self.callDown)
            self.elevatorList.append(elevator)

        if show_gui:
            self.ui_manager = GuiManager(self.elevatorList, self.callUp,
                                         self.callDown)
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
                waiting_time = list()
                for elevator in self.elevatorList:
                    elevator.operate()
                    waiting_time = waiting_time + elevator.waitingList
                self.avgWaitingTime.append(np.mean(waiting_time))
                Clock.tact += 1
                Logger.log()

            if self.showGui:
                self.ui_manager.render()

            Clock.tactBuffer = 0

            Clock.deltaTime = (t_new - t_old) * Clock.speedScale if self.showGui else 1
            t_old = t_new
            if not Clock.pause:
                Clock.add_time(Clock.deltaTime)

        self.shutdown()

    def shutdown(self):
        Clock.end_of_day = True
        data = self.end_of_day_log()
        print(data)
        if Conf.showPlots:
            self.draw_data(data)

    def end_of_day_log(self):
        # TODO: EOD Log verbessern für Reinforcement und einmal für Log Ordner
        Logger.new_tact()
        log = self.personManager.end_of_day()
        for elevator in self.elevatorList:
            log = log | elevator.end_of_day()

        waitingList = []
        for i in range(len(self.elevatorList)):
            log[f"waitingTime{i}"] = self.elevatorList[i].waitingList
            waitingList.extend(self.elevatorList[i].waitingList)
        if waitingList:
            log["avgWaitingTime"] = np.mean(waitingList)
        else:
            log["avgWaitingTime"] = None

        Logger.add_data(log)
        Logger.log()
        return log

    def draw_data(self, data: dict):
        # gamma
        # Histogramm erstellen
        fig, ax = plt.subplots(layout='constrained')
        plt.hist(self.personManager.scheduleTimes, bins=24 * 60, density=True, alpha=0.7)
        # Achsenbeschriftungen
        plt.xlabel('Zeit [Minuten]')
        plt.ylabel('Dichte')
        min_to_hour = lambda x: np.divide(x, 60)
        secax = ax.secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
        secax.set_xlabel('Zeit [Stunden]')
        # Titel des Plots
        plt.title('Gamma-Verteilung')
        # Diagramm anzeigen
        Logger.log('Gamma-Verteilung')
        plt.show()

        # Etagen
        floors = list()
        for f in self.personManager.homeFloors:
            floors.append(int(f) + 1)
        # Histogramm erstellen
        plt.hist(floors, bins=Conf.maxFloor, density=True, alpha=0.7)
        # Achsenbeschriftungen
        plt.xlabel('Etagen')
        plt.ylabel('Dichte')
        # Titel des Plots
        plt.title('Stockwerk-Verteilung')
        # Diagramm anzeigen
        Logger.log(plot_name='Etagen-Verteilung')
        plt.show()

        # Wartezeit
        y_all = range(len(self.avgWaitingTime))
        # Histogramm erstellen
        plt.hist(self.avgWaitingTime, bins=y_all, density=True, alpha=0.7)
        # Achsenbeschriftungen
        # fig, ax = plt.subplots(layout='constrained')
        plt.xlabel('Zeit [Minuten]')
        # secax = ax.secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
        # secax.set_xlabel('Zeit [Stunden]')
        plt.ylabel('Wartezeit [Minuten]')
        # Titel des Plots
        plt.title('Durchschnittliche Wartezeiten')
        # Diagramm anzeigen
        Logger.log(plot_name='Durchschnittliche-Wartezeit')
        plt.show()


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
