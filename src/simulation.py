import sys
from time import time
import numpy as np
from matplotlib import pyplot as plt

from src.logic.person import Person
from src.utils import Conf, Clock, Logger
from src.logic.elevator import Elevator
from src.logic.person_manager import PersonManager
from src.ui.gui_manager import GuiManager


class Simulation:
    avgWaitingTime: list[np.ndarray]
    callUp: list[list[Person]]
    callDown: list[list[Person]]

    personManager: PersonManager
    elevators: list[Elevator]

    showGui: bool

    def __init__(self, show_gui=True, elevator_position: tuple[int, int, int] = (0, 0, 0)):
        self.showGui = show_gui
        self.avgWaitingTime = list()
        self.callUp = [list() for _ in range(Conf.maxFloor)]
        self.callDown = [list() for _ in range(Conf.maxFloor)]

        self.personManager = PersonManager(self.callUp,
                                           self.callDown)

        self.elevators = list()
        for i in range(3):
            elevator = Elevator(call_up=self.callUp, call_down=self.callDown,
                                start_position=elevator_position[i])
            self.elevators.append(elevator)

        if show_gui:
            self.ui_manager = GuiManager(self.elevators, self.callUp,
                                         self.callDown)
        self.run()

    def run(self) -> None:
        """
        Update and render loop of the simulation.
        :return: None
        """
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
                #
                # calculate average waiting time
                elevator_waiting_time: list[tuple[int, int]] = list()
                for elevator in self.elevators:
                    elevator.manage()
                    elevator_waiting_time = elevator_waiting_time + elevator.waitingTimes
                if not elevator_waiting_time:
                    elevator_waiting_time.append((Clock.tact, 0))
                avg_waiting_time = np.mean(elevator_waiting_time, axis=0)[1]
                avg_waiting_time = np.nan_to_num(avg_waiting_time, nan=0.0)
                self.avgWaitingTime.append(avg_waiting_time)

                Logger.add_data({'avgWaitingTime': avg_waiting_time})
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
        print("Average Waiting Time:", f"{data['avgWaitingTime']:.2f}", " | Remaining Persons:", data["remaining"])
        if Conf.showPlots:
            self.draw_data(data)

    def end_of_day_log(self):
        # TODO: End of day
        Logger.new_tact()
        log = self.personManager.end_of_day()
        for elevator in self.elevators:
            log = log | elevator.end_of_day()

        elevator_waiting_time = []
        for i in range(len(self.elevators)):
            log[f"waitingTime{i}"] = self.elevators[i].waitingTimes
            elevator_waiting_time.extend(self.elevators[i].waitingTimes)
        if elevator_waiting_time:
            log["avgWaitingTime"] = self.avgWaitingTime[len(self.avgWaitingTime) - 1]
        else:
            log["avgWaitingTime"] = None

        Logger.add_data(log)
        Logger.log()
        return log

    def draw_data(self, data: dict) -> None:
        """
        Drawing data to plots
        :param data: dictionary of log data
        :return: None
        """
        # gamma
        #
        fig, axs = plt.subplots(layout='constrained')
        plt.hist(self.personManager.scheduleTimes, bins=24 * 60, density=True)

        plt.xlabel('Zeit [Minuten]')
        plt.ylabel('Dichte')
        min_to_hour = lambda x: np.divide(x, 60)
        secax1 = axs.secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
        secax1.set_xlabel('Zeit [Stunden]')

        plt.title('Gamma-Verteilung')

        Logger.log('Gamma-Verteilung')
        plt.show()

        # floors
        #
        floors = list()
        for f in self.personManager.homeFloors:
            floors.append(int(f) + 1)
        # Histogramm erstellen
        plt.hist(floors, bins=[i for i in range(1, Conf.maxFloor + 1)], density=True, alpha=0.8)
        # Achsenbeschriftungen
        plt.xlabel('Etagen')
        plt.ylabel('Dichte')
        # Titel des Plots
        plt.title('Stockwerk-Verteilung')
        # Diagramm anzeigen
        Logger.log(plot_name='Etagen-Verteilung')
        plt.show()

        # average waiting time
        #
        fig, axs = plt.subplots(2, layout='constrained')
        axs[1].plot([i for i in range(24 * 60)], self.avgWaitingTime)
        plt.xlabel('Zeit [Minuten]')
        secax1 = axs[1].secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
        secax1.set_xlabel('Zeit [Stunden]')
        plt.ylabel('Wartezeit [Minuten]')

        axs[0].plot([i for i in range(24 * 60)], self.personManager.numberInMotion)
        plt.xlabel('Zeit [Minuten]')
        secax2 = axs[0].secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
        secax2.set_xlabel('Zeit [Stunden]')
        plt.ylabel('Wartezeit [Minuten]')

        fig.suptitle('Durchschnittliche Wartezeiten')
        plt.title('Reisende Personen')

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
        elif param.__contains__("skip="):
            value = param[5:]
            Clock.skip = int(value)

    S = Simulation(show_gui=showGui)
