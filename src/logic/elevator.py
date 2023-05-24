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

    def __init__(self, call_up, call_down, start_position=0):
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

    def manage(self) -> None:
        """
        Manages the elevator each tact
        :return: None
        """
        self.state = self.nextState

        # if at buildings end change direction
        if self.position == Conf.maxFloor - 1:
            self.direction = Direction.DOWN
        elif self.position == 0:
            self.direction = Direction.UP

        log: dict = dict()

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
                target_floor = Conf.maxFloor - 1 if self.direction == Direction.UP else 0
                for p in self.passengers:
                    floor = p.schedule[0][1]
                    if self.direction == Direction.UP:
                        if floor <= target_floor:
                            self.nextState = ElevatorState.UP
                            target_floor = floor
                    else:
                        if floor >= target_floor:
                            self.nextState = ElevatorState.DOWN
                            target_floor = floor
                if self.nextState != ElevatorState.WAIT:
                    # if passengers with target exist, set new target
                    self.target = target_floor
        else:
            if len(self.passengers) == 0 and self.position == self.target:
                # if no passengers doesn't have a target
                target_floor = self.search_for_call()
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

        # log elevator's variables
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

    def search_for_call(self) -> int | None:
        """
        Searchs the next requested floor from the current position following the elevator's direction.
        If building end reached turn direction and search again.
        :return: Requested floor or None if no requests exists
        """
        call: list[Person]

        searched_one_direction: bool = False

        for _ in range(2):
            # 4 cases per direction:
            #
            # if on the way up: check if anyone from position to 14 wants to go up.
            # if on the way up: check if anyone from 14 to position wants to go down.
            # switch direction: up -> down
            # if on the way up: check if anyone from position to 0 wants to go down.
            # if on the way up: check if anyone from 0 to position wants to go up.
            #
            # if on the way down: check if anyone from position to 0 wants to go down.
            # if on the way down: check if anyone from 0 to position wants to go up.
            # switch direction: down -> up
            # if on the way down: check if anyone from position to 14 wants to go up.
            # if on the way down: check if anyone from 14 to position wants to go down.

            # if already on the buildings end, only one direction have to be checked
            if self.position == Conf.maxFloor - 1:
                searched_one_direction = True
            elif self.position == 0:
                searched_one_direction = True

            search_range = (range(self.position, Conf.maxFloor), range(0, self.position + 1))
            if self.direction == Direction.UP:
                for i in search_range[0]:  # position -> 14
                    if self.callUp[i]:
                        return i
                for i in reversed(search_range[0]):  # 14 -> position
                    if self.callDown[i]:
                        # if current floor requested follow the request to go down
                        if self.position == i:
                            self.direction = Direction.DOWN
                        return i
                # change direction if requested: up -> down
                for i in reversed(search_range[1]):  # position -> 0
                    if self.callDown[i]:
                        self.direction = Direction.DOWN
                        return i
                for i in search_range[1]:  # 0 -> position
                    if self.callUp[i]:
                        self.direction = Direction.DOWN
                        return i
            else:
                for i in reversed(search_range[1]):  # position -> 0
                    if self.callDown[i]:
                        return i
                for i in search_range[1]:  # 0 -> pos
                    if self.callUp[i]:
                        # if current floor requested follow the request to go up
                        if self.position == i:
                            self.direction = Direction.UP
                        return i
                # change direction if requested: down -> up
                for i in search_range[0]:  # pos -> 14
                    if self.callUp[i]:
                        self.direction = Direction.UP
                        return i
                for i in reversed(search_range[0]):  # 14 -> pos
                    if self.callDown[i]:
                        self.direction = Direction.UP
                        return i

            if searched_one_direction:
                break
            else:
                self.direction = Direction.UP if self.direction == Direction.DOWN else Direction.DOWN
                searched_one_direction = True
        return None

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

    def leaving_passengers(self) -> None:
        """
        Removes people who are on their target floor and set there waitingStartTime to None.
        :return: None
        """
        if self.state == ElevatorState.WAIT:
            for person in self.passengers:
                if person.schedule[0][1] == self.position:
                    person.schedule.pop(0)
                    assert person.waitingStartTime
                    self.waitingTimes.append((Clock.tact, Clock.tact - person.waitingStartTime))

                    person.waitingStartTime = None
                    person.position = self.position
                    self.passengers.remove(person)

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
