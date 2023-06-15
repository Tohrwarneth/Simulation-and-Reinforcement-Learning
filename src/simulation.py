import argparse
from time import time
import numpy as np
from matplotlib import pyplot as plt

from logic.person import Person
from utils import Conf, Clock, Logger
from logic.elevator import Elevator
from logic.person_manager import PersonManager
from ui.gui_manager import GuiManager


class Simulation:
    avgWaitingTime: list[np.ndarray]
    finalAvgWaitingTime: list[np.ndarray]
    callUp: list[list[Person]]
    callDown: list[list[Person]]

    personManager: PersonManager
    elevators: list[Elevator]

    show_gui: bool

    def __init__(self, show_gui=True, rl_decider: bool = False,
                 elevator_position: tuple[int, int, int] = (0, 0, 0)):
        self.showGui = show_gui
        self.avgWaitingTime = list()
        self.finalAvgWaitingTime = list()
        self.callUp = [list() for _ in range(Conf.maxFloor)]
        self.callDown = [list() for _ in range(Conf.maxFloor)]

        self.personManager = PersonManager(self.callUp,
                                           self.callDown)

        self.elevators = list()
        for i in range(3):
            elevator = Elevator(call_up=self.callUp, call_down=self.callDown,
                                start_position=elevator_position[i])
            if rl_decider:
                elevator.decider = Conf.reinforcement_decider
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
        while Clock.running or Clock.tactBuffer > 0:
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
                person_waiting_time: list[int] = list()
                #
                # finished when average waiting time for end of day
                for elevator in self.elevators:
                    elevator.manage()
                    elevator_waiting_time += elevator.waitingTimes
                if len(elevator_waiting_time) == 0:
                    elevator_waiting_time.append((Clock.tact, 0))
                avg_waiting_time = np.mean(elevator_waiting_time, axis=0)[1]
                self.finalAvgWaitingTime.append(np.nan_to_num(avg_waiting_time, nan=0.0))
                #
                # current average waiting time
                for person in self.personManager.persons:
                    if person.waitingStartTime:
                        person_waiting_time.append(Clock.tact - person.waitingStartTime)
                if len(person_waiting_time) == 0:
                    person_waiting_time.append(0)
                avg_waiting_time = np.mean(person_waiting_time)
                self.avgWaitingTime.append(avg_waiting_time)

                Clock.tact += 1

                if not Conf.train:
                    Logger.add_data({'avgWaitingTime': avg_waiting_time})
                    Logger.log()

            if self.showGui:
                self.ui_manager.render()

            Clock.tactBuffer = 0

            Clock.deltaTime = (t_new - t_old) * Clock.speedScale if self.showGui else 1
            t_old = t_new
            if not Clock.pause:
                Clock.add_time(Clock.deltaTime)

        self.shutdown()

    def shutdown(self) -> None:
        """
        Called after simulation finished
        :return: None
        """
        Clock.endOfDay = True
        data = self.end_of_day_log()
        print(f"Average Waiting Time: {data['avgWaitingTime']:.2f} | "
              f"Remaining Persons: {data['remaining']}")
        if Conf.generatesPlots:
            self.draw_data()

    def end_of_day_log(self) -> dict:
        """
        Returns final log in dictionary
        :return: log dictionary
        """
        Logger.new_tact()
        log: dict = dict()

        if self.finalAvgWaitingTime:
            avg_waiting = self.finalAvgWaitingTime.pop()
            log["avgWaitingTime"] = avg_waiting
        else:
            log["avgWaitingTime"] = None

        log = log | self.personManager.end_of_day()
        for elevator in self.elevators:
            log = log | elevator.end_of_day()

        Logger.add_data(log)
        Logger.log()
        return log

    def draw_data(self) -> None:
        """
        Drawing data to plots
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
        if Conf.showPlots:
            plt.show()

        # floors
        #
        floors = list()
        for f in self.personManager.homeFloors:
            floors.append(int(f) + 1)

        plt.hist(floors, bins=[i for i in range(1, Conf.maxFloor + 1)], density=True, alpha=0.8)

        plt.xlabel('Etagen')
        plt.ylabel('Dichte')

        plt.title('Stockwerk-Verteilung')

        Logger.log(plot_name='Etagen-Verteilung')
        if Conf.showPlots:
            plt.show()

        fig: plt.Figure = plt.figure()
        ax_motion: plt.Axes = fig.add_subplot(211)
        ax_avg: plt.Axes = fig.add_subplot(212)
        ax_motion.plot([i for i in range(24 * 60)], self.personManager.numberInMotion)
        ax_motion.set_xlabel('Zeit [Minuten]')
        ax_motion.set_ylabel('Personen')
        secax0 = ax_motion.secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
        secax0.set_xlabel('Zeit [Stunden]')
        ax_motion.title.set_text('Reisende Personen')

        ax_avg.plot([i for i in range(24 * 60)], self.avgWaitingTime)
        ax_avg.set_xlabel('Zeit [Minuten]')
        secax1 = ax_avg.secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
        secax1.set_xlabel('Zeit [Stunden]')
        ax_avg.set_ylabel('Wartezeit [Minuten]')
        fig.suptitle("Durchschnittliche Wartezeit")

        fig.tight_layout()

        Logger.log(plot_name='Durchschnittliche-Wartezeit')
        if Conf.showPlots:
            fig.show()

        # average waiting time
        #
        fig: plt.Figure = plt.figure()
        ax_avg: plt.Axes = fig.add_subplot(211)
        ax_final_avg: plt.Axes = fig.add_subplot(212)

        ax_avg.plot([i for i in range(24 * 60)], self.avgWaitingTime)
        ax_avg.set_xlabel('Zeit [Minuten]')
        secax1 = ax_avg.secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
        secax1.set_xlabel('Zeit [Stunden]')
        ax_avg.set_ylabel('Wartezeit [Minuten]')
        fig.suptitle("Durchschnittliche Wartezeit")

        ax_final_avg.plot([i for i in range(24 * 60)], self.finalAvgWaitingTime)
        ax_final_avg.set_xlabel('Zeit [Minuten]')
        secax2 = ax_final_avg.secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
        secax2.set_xlabel('Zeit [Stunden]')
        ax_final_avg.set_ylabel('Wartezeit [Minuten]')
        ax_final_avg.title.set_text('Finale Durchschnittliche Wartezeit')
        fig.tight_layout()
        if Conf.showPlots:
            fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ElevatorSimulation',
                    description='Simulates elevators of an office complex in a simple way')

    parser.add_argument('-p', '--plots', help="Generates plots", action='store_true')
    parser.add_argument('-sp', '--showPlots', help="Shows generated plots. (sets --plots true)", action='store_true')
    parser.add_argument('-ui', '--ui', help="Shows user interface", action='store_true')
    parser.add_argument('-s', '--skip', help="Fast forward to hour x", type=int, nargs='?')
    parser.add_argument('-rl', '--reinforcementLearner',
                        help="Runs the simulation with reinforcement learned Decider", action='store_true')
    parser.add_argument('-t', '--train', help="Trains the reinforcement learner", action='store_true')
    parser.add_argument('-nl', '--noLogs', help="Doesn't generates log files", action='store_true')

    args = parser.parse_args()
    argument_dict: dict = vars(args)
    print(f"Dict format: {argument_dict}")

    show_gui: bool = args.ui
    Conf.generatesPlots = args.plots if args.plots or args.showPlots else False
    Conf.showPlots = args.showPlots
    Clock.skip = args.skip
    reinforcement_learning: bool = args.reinforcementLearner
    Conf.train = args.train
    Logger.noLogs = args.noLogs

    S = Simulation(show_gui=show_gui, rl_decider=reinforcement_learning)
