from __future__ import annotations
from time import time
import numpy as np
import torch
from matplotlib import pyplot as plt

import enums
from logic.person import Person
from re_learner import NetCoder
import re_learner.TrainerPPO
import re_learner.Net
from re_learner.reward import Reward
from utils import Conf, Clock, Logger
from logic.elevator import Elevator
from logic.person_manager import PersonManager
from ui.gui_manager import GuiManager


class Simulation:
    avgWaitingTime: list[float] | list[np.ndarray]
    latestAvgWaitingTime: float
    finalAvgWaitingTime: list[float] | list[np.ndarray]
    callUp: list[list[Person]]
    callDown: list[list[Person]]

    personManager: PersonManager
    elevators: list[Elevator]

    showGui: bool
    rlDecider: bool
    latestDecision: tuple | None
    reward: float = 0
    rewardList: list[float] = list()

    def __init__(self, show_gui=True, rl_decider: bool = False,
                 elevator_position: tuple[int, int, int] = (0, 0, 0)):
        self.latestDecision = None
        self.showGui = show_gui
        self.rlDecider = rl_decider

        self.avgWaitingTime = list()
        self.latestAvgWaitingTime = 0.0
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
        if rl_decider:
            NetCoder.init(self.get_game_state(), Conf.capacity)

    def run(self, episode_buffer: list[tuple, float, int] = None) -> None:
        """
        Update and render loop of the simulation.
        :return: None
        """
        Logger.init()

        if Clock.skip:
            Clock.speedScale = 265

        t_old: float = time()
        while Clock.running or Clock.tactBuffer > 0:
            self.reward = 0
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
                need_decisions = [False, False, False]
                for i, elevator in enumerate(self.elevators):
                    need_decisions[i], local_reward = elevator.manage()
                    self.reward += local_reward
                    elevator_waiting_time += elevator.waitingTimes
                    if not self.rlDecider:
                        elevator.log()
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
                self.latestAvgWaitingTime = self.avgWaitingTime[len(self.avgWaitingTime) - 1]

                if self.rlDecider:
                    # Nicht, wenn auf Warten gestellt wird. Also nur, wenn er Entscheidung treffen soll
                    decisions = Conf.reinforcement_decider.get_decision(self)
                    self.apply_decisions(decisions, need_decisions)
                    self.rewardList.append(self.reward)

                Clock.tact += 1

                if not Conf.train:
                    Logger.add_data({'avgWaitingTime': avg_waiting_time})
                    Logger.log()

                if episode_buffer is not None:
                    episode_buffer.append((self.get_game_state(), self.latestDecision, self.reward))

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
        if not Conf.train:
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

    def get_game_state(self) -> tuple:
        elevators: list[tuple] = list()
        for e in self.elevators:
            elevators.append((e.position, e.direction, e.nextState, e.passengers))
        if len(self.avgWaitingTime) == 0:
            avg_waiting = 0.0
        else:
            avg_waiting = self.latestAvgWaitingTime

        return Clock.tact, avg_waiting, \
            self.personManager.get_remaining_people(), \
            self.callUp, self.callDown, \
            tuple(elevators)

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

    def apply_decisions(self, decisions: tuple[enums.ElevatorState, enums.ElevatorState, enums.ElevatorState],
                        need_decisions: list[bool, bool, bool]) -> None:
        self.latestDecision = decisions
        self.reward -= self.latestAvgWaitingTime
        # self.reward += self.personManager.get_people_in_motion() * Reward.notHomePenalty
        for index, elevator in enumerate(self.elevators):
            if need_decisions[index]:
                elevator.apply_decision(decisions[elevator.index])
            elevator.log()

    @staticmethod
    def load_rl_model(model_file: str, device: torch.device) -> re_learner.Net.Net:
        try:
            net = torch.load(model_file)
            net.to(device)
        except:
            net = re_learner.Net.Net()
        Conf.reinforcement_decider.init(net)
        return net

    @staticmethod
    def reset():
        Elevator.nextElevatorIndex = 0
        Person.nextPersonIndex = 0
        Clock.reset()


if __name__ == "__main__":
    show_gui, reinforcement_learning = Conf.parse_args()

    if Conf.train:
        re_learner.TrainerPPO.TrainerPPO().TrainNetwork()
    else:
        sim = Simulation(show_gui=show_gui, rl_decider=reinforcement_learning)
        if reinforcement_learning:
            Simulation.load_rl_model(re_learner.TrainerPPO.TrainerPPO.modelFile)
        sim.run()
        Simulation.reset()
