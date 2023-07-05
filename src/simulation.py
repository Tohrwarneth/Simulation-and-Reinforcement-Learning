from __future__ import annotations
from time import time
import numpy as np
import torch
from matplotlib import pyplot as plt

import enums
import plotter
from logic.person import Person
from new_re_learner import trainer, agent
from re_learner import NetCoder
import re_learner.TrainerPPO
import re_learner.Net
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
    peopleInOffice: int = 0
    reward: float = 0.0
    rewardList: list[float] = list()
    agent: agent.Agent = None

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

    def run(self, episode_buffer: list[tuple, float, int] = None, action: int = None, step: bool = False) -> \
            tuple[tuple, float, bool] | None:
        """
        Update and render loop of the simulation.
        :return: None
        """
        Logger.init()

        if Clock.skip:
            Clock.speedScale = 265

        run_step = True

        t_old: float = time()
        while (Clock.running or Clock.tactBuffer > 0) and run_step:
            self.reward = 0
            t_new: float = time()
            for _ in range(Clock.tactBuffer):
                if not Conf.train and Conf.phill:
                    observation = self.get_game_state()
                    action, _, _ = self.agent.choose_action(NetCoder.normalize_game_state(observation))
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
                action_decisions = NetCoder.decision_to_states(action)
                debug_position = list()
                for i, elevator in enumerate(self.elevators):
                    need_decisions[i], local_reward = elevator.manage(action_decisions[i])
                    self.reward += local_reward
                    elevator_waiting_time += elevator.waitingTimes
                    if not Logger.noLogs:
                        elevator.log()
                    if Logger.noLogs:
                        debug_position.append(elevator.position + 1)
                if Logger.noLogs and not Conf.train:
                    print(debug_position)
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

                if self.rlDecider and action is None:
                    # Nicht, wenn auf Warten gestellt wird. Also nur, wenn er Entscheidung treffen soll
                    decisions = Conf.reinforcement_decider.get_decision(self)
                    self.apply_decisions(decisions, need_decisions)
                current_people_in_office = self.personManager.get_people_in_office()

                if self.rlDecider:
                    # self.reward = -self.latestAvgWaitingTime
                    # self.reward -= self.latestAvgWaitingTime
                    # self.reward += Conf.totalAmountPerson - self.personManager.get_remaining_people()
                    current_people_in_office = self.personManager.get_people_in_office()

                    # self.reward = max(0, current_people_in_office - self.peopleInOffice)
                    self.rewardList.append(self.reward)

                    self.peopleInOffice = current_people_in_office

                Clock.tact += 1

                if not Conf.train:
                    Logger.add_data({'avgWaitingTime': avg_waiting_time})
                    Logger.log()

                if episode_buffer is not None and action is None:
                    episode_buffer.append((self.get_game_state(), self.latestDecision, self.reward))

            if self.showGui:
                self.ui_manager.render()

            Clock.tactBuffer = 0

            Clock.deltaTime = (t_new - t_old) * Clock.speedScale if self.showGui else 1
            t_old = t_new
            if not Clock.pause:
                Clock.add_time(Clock.deltaTime)

            if step:
                # if call run, it goes step by step
                run_step = False

        if step:
            # state, reward, done (while run condition)
            return self.get_game_state(), self.reward, not (Clock.running or Clock.tactBuffer > 0)
        else:
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
            plotter.draw_data(self)

    def end_of_day_log(self) -> dict:
        """
        Returns final log in dictionary
        :return: log dictionary
        """
        Logger.new_tact()
        log: dict = dict()

        if self.finalAvgWaitingTime:
            # avg_waiting = self.finalAvgWaitingTime.pop()
            log["avgWaitingTime"] = self.latestAvgWaitingTime
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

    def apply_decisions(self, decisions: tuple[enums.ElevatorState, enums.ElevatorState, enums.ElevatorState],
                        need_decisions: list[bool, bool, bool]) -> None:
        self.latestDecision = decisions
        # self.reward += self.personManager.get_people_in_motion() * Reward.notHomePenalty
        for index, elevator in enumerate(self.elevators):
            if need_decisions[index]:
                elevator.apply_decision(decisions[elevator.index])
            elevator.log()

    @staticmethod
    def load_rl_model(model_file: str, device: torch.device, is_phill=False,
                      sim: Simulation = None) -> re_learner.Net.Net:
        if is_phill:
            return trainer.PPOTrainer.get_new_actor(sim)
        else:
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

    def sim_reset(self):
        self.reset()
        self.__init__(self.showGui, self.rlDecider)
        return self.get_game_state()


if __name__ == "__main__":
    show_gui, reinforcement_learning, rl_phill, rl_dice = Conf.parse_args()

    if Conf.train:
        if rl_dice:
            re_learner.TrainerPPO.TrainerPPO().TrainNetwork()
        elif rl_phill:
            trainer.PPOTrainer().train()
    else:
        sim = Simulation(show_gui=show_gui, rl_decider=reinforcement_learning)
        if reinforcement_learning:
            device = torch.device('cpu')
            if rl_dice:
                Simulation.load_rl_model(re_learner.TrainerPPO.TrainerPPO.modelFile, device)
            elif rl_phill:
                sim.agent = Simulation.load_rl_model(trainer.PPOTrainer.modelFile, device, rl_phill, sim)
        sim.run()
        Simulation.reset()
