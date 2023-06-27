from logic.decider_interface import IDecider
from logic.decision import Decision
from logic.simulation_decider import SimulationDecider
from re_learner.reward import Reward
from utils import Conf, Clock, Logger
from enums import ElevatorState, Direction
from logic.person import Person


class Elevator:
    """
    Model and logic of a single elevator
    """
    nextElevatorIndex: int = 0
    index: int

    passengers: list[Person]
    state: ElevatorState
    nextState: ElevatorState  # future state for the next tact
    direction: Direction
    position: int
    target: int

    callDown: list[list[Person]]
    callUp: list[list[Person]]
    waitingTimes: list[tuple[int, int]]
    #
    # Variables described in config class
    capacity: int
    decider: IDecider
    decision_job: tuple[int, Direction] | None
    decision_call: int | None

    reward: float = 0

    def __init__(self, call_up, call_down, start_position=0, decider: IDecider = SimulationDecider):
        self.index = self.nextElevatorIndex
        Elevator.nextElevatorIndex += 1
        self.passengers = list()
        self.state = ElevatorState.WAIT
        self.nextState = ElevatorState.WAIT
        self.direction = Direction.UP
        self.position = start_position
        self.target = start_position
        #
        self.callDown = call_down
        self.callUp = call_up
        self.waitingTimes = list()
        #
        self.capacity = Conf.capacity
        self.decider = decider

    def manage(self, decision=None) -> tuple[bool, float]:
        """
        Manages the elevator each tact
        :return: if a decision for Reinforcement Learner is needed and reward of the tact
        """
        self.reward = 0
        self.state = self.nextState
        need_decision: bool = False

        # if at buildings end change direction
        if self.position == Conf.maxFloor - 1:
            self.direction = Direction.DOWN
        elif self.position == 0:
            self.direction = Direction.UP

        # TODO: schauen ob erste Zeile len(self.passengers) > 0 nichts kaputt macht
        if (self.target == self.position and len(self.passengers)) or (
                self.is_floor_requested() and len(self.passengers) < self.capacity):
            # job done or elevator is requestes while driving
            if self.state != ElevatorState.WAIT:
                # if reached target floor, wait a tact
                self.nextState = ElevatorState.WAIT
            else:
                # Elevator is waiting for passengers to leave and enter
                self.leaving_passengers()
                self.entering_passengers()

                # reevaluate the new target floor
                target_floor, next_state = self.decider.get_next_job(self.position, self.direction, self.nextState,
                                                                     self.passengers)
                need_decision = True
                if target_floor != None:
                    self.target = target_floor
                    self.nextState = next_state

                # rl decision overwrites sim decision
                if decision is not None:
                    self.apply_decision(decision)
        else:
            if len(self.passengers) == 0 and self.position == self.target:
                # if no passengers and doesn't have a target
                target_floor, self.direction = self.decider.search_for_call(position=self.position,
                                                                            direction=self.direction,
                                                                            call_up=self.callUp,
                                                                            call_down=self.callDown)
                need_decision = True
                if target_floor != None:
                    # is requested
                    self.target = target_floor
                    if self.target == self.position:
                        self.nextState = ElevatorState.WAIT
                    else:
                        self.nextState = ElevatorState.UP if self.direction == Direction.UP else ElevatorState.DOWN
                else:
                    # wait if not requested
                    self.nextState = ElevatorState.WAIT

                # rl decision overwrites sim decision
                if decision is not None:
                    self.apply_decision(decision)
            if self.position != self.target:
                # if driving to a target
                if len(self.passengers) > 0:
                    # state represent direction of the elevator
                    self.nextState = ElevatorState.UP if self.direction == Direction.UP else ElevatorState.DOWN
                # position is changed by one floor
                self.position += self.state.value
                if self.position == Conf.maxFloor:
                    self.position = Conf.maxFloor - 1
                if self.position == -1:
                    self.position = 0
        return need_decision, self.reward

    def log(self):
        # log elevator's variables
        log: dict = dict()
        log[f"({self.index}) position"] = self.position
        log[f"({self.index}) target"] = self.target
        log[f"({self.index}) state"] = self.state
        log[f"({self.index}) number of passangers"] = len(self.passengers)
        log[f"({self.index}) passangers"] = self.passengers
        Logger.add_data(log)

    def is_floor_requested(self) -> bool:
        """
        Checks if the elevator's direction of the current floor is requested
         or if it's a passengers target floor.
        :return: True if the current floor ist requested
        """
        call: list[Person]
        if self.direction == Direction.UP:
            call = self.callUp[self.position]
        else:
            call = self.callDown[self.position]
        return call or any(p.schedule[0][1] == self.position for p in self.passengers)

    def entering_passengers(self) -> None:
        """
        Adds passengers according to the call direction and the direction of the elevator.
        :return: None
        """
        if self.state == ElevatorState.WAIT:
            call: list[Person]

            if self.direction == Direction.UP:
                call = self.callUp[self.position]
            else:
                call = self.callDown[self.position]

            while len(self.passengers) < self.capacity and call:
                p = call.pop(0)
                self.passengers.append(p)
                self.reward += Reward.enterReward

    def leaving_passengers(self) -> None:
        """
        Removes people who are on their target floor and set there waitingStartTime to None.
        :return: None
        """
        if self.state == ElevatorState.WAIT:
            for person in self.passengers:
                if person.schedule[0][1] == self.position:
                    person.schedule.pop(0)
                    assert person.waitingStartTime is not None
                    self.waitingTimes.append((Clock.tact, Clock.tact - person.waitingStartTime))

                    person.waitingStartTime = None
                    person.position = self.position
                    self.passengers.remove(person)
                    self.reward += Reward.leaveReward

    def end_of_day(self) -> dict:
        """
        Returns final log in dictionary
        :return: log dictionary
        """
        log: dict = dict()
        log[f"({self.index}) position"] = self.position
        log[f"({self.index}) number of passangers"] = len(self.passengers)
        log[f"({self.index}) passangers"] = self.passengers
        return log

    def __repr__(self):
        return f"index: {self.index}, position: {self.position}, target: {self.target}," \
               f" passengers: {len(self.passengers)}, state: {self.state}, next state: {self.nextState}"

    def apply_decision(self, decision: ElevatorState):
        new_target = self.target + decision.value

        if new_target > 14:
            new_target = 14
            decision = ElevatorState.WAIT
        elif new_target < 0:
            new_target = 0
            decision = ElevatorState.WAIT

        self.direction = Direction.UP if self.target >= new_target else Direction.DOWN
        self.target = new_target

        self.nextState = decision
        return self.reward

        #
        # if self.target == self.position:
        #     self.nextState = ElevatorState.WAIT
        # else:
        #     self.nextState = ElevatorState.UP if self.direction == Direction.UP else ElevatorState.DOWN
