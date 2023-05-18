from src.utils import Conf, Clock, Logger
from src.enums import ElevatorState, Direction
from src.logic.person import Person


class Elevator:
    nextElevatorIndex: int = 0
    index: int
    passengers: list[Person]
    state: ElevatorState
    nextState: ElevatorState  # Für die GUI
    position: int
    target: int

    # TODO: zu job Ziel fahren und wenn auf dem Weg jmd. ist und Platz ist, einpacken.
    #  Wenn nix zu tun, schau zuerst in deine ursprüngliche Richtung weiter, wenn da nix ist, dann unter dir.

    def __init__(self, call_up, call_down, capacity, speed=1, waitingTime=1, startPos=0):
        self.index = self.nextElevatorIndex
        Elevator.nextElevatorIndex += 1
        self.target = 0
        self.state = ElevatorState.WAIT
        self.nextState = ElevatorState.WAIT
        self.direction = Direction.UP
        # private var
        self.callDown = call_down
        self.callUp = call_up
        self.waitingList: list[int] = []
        self.passengers = []
        self.capacity = capacity
        self.speed = speed
        self.position = startPos
        self.waitingTime = waitingTime

    # def isPassengerLeaving(self):
    #     '''
    #     checks if a passenger wants to leave on the current floor
    #     Returns:
    #
    #     '''
    #     result = False
    #     for p in self.passengers:
    #         if p.schedule[0][1] == self.position:
    #             result = True
    #
    #     return result

    def personsLeaving(self):
        '''
        Iterates over passengers
        sets there new location
        removes task from schedule
        resets there startwaitingTime = None
        accumulates the waitingTime in waitingList

        '''
        for p in self.passengers:
            if p.schedule[0][1] == self.position:  # first task, location
                p.schedule.pop(0)  # schedule task finished
                self.waitingList.append(Clock.tact - p.startWaitingTime)

                # Personscontroller can see that the person is no longer waiting
                p.startWaitingTime = None
                p.location = self.position
                self.passengers.remove(p)

    def personsEntering(self):
        '''
        adds passengers till capacity is reached or no people on the floor
        people going in the same direction as the elevator a prioritized
        '''
        if self.state == ElevatorState.WAIT:
            call: list[Person]

            if self.direction == Direction.UP:
                call = self.callUp[self.position]
            else:
                call = self.callDown[self.position]

            while len(self.passengers) < self.capacity and call:
                p = call.pop(0)
                self.passengers.append(p)

    def isFloorRequested(self):
        '''
        checks if the Floor at the current position is requested
        '''
        call: list[Person]
        if self.direction == Direction.UP:
            call = self.callUp[self.position]
        else:
            call = self.callDown[self.position]
        return call or any(p.schedule[0][1] == self.position for p in self.passengers)

    """
    # def isFloorRequestedUpwards(self):
    #     '''
    #     Returns: is a person waiting on the current floor to go upwards
    #     '''
    #     return (self.callUp[self.position]
    #             or any(p.schedule[0][1] == self.position for p in self.passengers))
    #
    # def isFloorRequestedDownwards(self):
    #     '''
    #     Returns: is a person waiting on the current floor to go downwards
    #     '''
    #     return (self.callDown[self.position]
    #             or any(p.schedule[0][1] == self.position for p in self.passengers))
    #
    # def isFloorAboveRequested(self):
    #     '''
    #     checks if a Floor above current position is requested
    #     '''
    #     result = False
    #     if any(p.schedule[0][1] > self.position for p in self.passengers):
    #         result = True
    #
    #     for i in range(self.position + 1, len(self.callUp)):
    #         if self.callUp[i] or self.callDown[i]:
    #             result = True
    #
    #     return result
    #
    # def isFloorBelowRequested(self):
    #     '''
    #     checks if a Floor below current position is Requested
    #     '''
    #     result = False
    #     if any(p.schedule[0][1] < self.position for p in self.passengers):
    #         result = True
    #
    #     for i in range(self.position):  # TODO numOfFloors as attribute
    #         if self.callDown[i] or self.callUp[i]:
    #             result = True
    #
    #     return result
    """

    def search_for_call(self) -> int | None:
        """
        Sucht den nächsten angeforderten Stock von aktueller Position aus.
        Wenn Gebäudegrenze erreicht wurde, wird die Richtung umgedreht.
        :return: Angefordertes Stockwerk oder None, wenn nichts gefunden
        """
        call: list[Person]
        search_range: tuple[range | reversed, range | reversed] | None = None
        searched_one_way: bool = False

        for _ in range(2):
            # 4 Fälle pro Richtung:
            #
            # wenn auf dem Weg nach oben: schauen, ob von pos bis 15 jemand nach oben will.
            # wenn auf dem Weg nach oben: schauen, ob von 15 bis pose jemand nach unten will.
            # wenn auf dem Weg nach oben: schauen, ob von pos bis 0 jemand nach unten will.
            # wenn auf dem Weg nach oben: schauen, ob von 0 bis pos jemand nach oben will.
            #
            # wenn auf dem Weg nach unten: schauen, ob von pos bis 0 jemand nach unten will.
            # wenn auf dem Weg nach unten: schauen, ob von 0 bis pos jemand nach oben will.
            # wenn auf dem Weg nach unten: schauen, ob von pos bis 15 jemand nach oben will.
            # wenn auf dem Weg nach unten: schauen, ob von 15 bis 0 jemand nach unten will

            if self.position == Conf.maxFloor - 1:
                searched_one_way = True
            elif self.position == 0:
                searched_one_way = True

            search_range = (range(self.position, Conf.maxFloor), range(0, self.position + 1))
            if self.direction == Direction.UP:
                for i in search_range[0]:  # pos -> 14
                    if self.callUp[i]:
                        return i
                for i in reversed(search_range[0]):  # 14 -> pos
                    if self.callDown[i]:
                        if self.position == i:
                            self.direction = Direction.DOWN
                        return i
                # Nach unten schauen und ggf. Richtung wechseln
                for i in reversed(search_range[1]):  # pos -> 0
                    if self.callDown[i]:
                        self.direction = Direction.DOWN
                        return i
                for i in search_range[1]:  # 0 -> pos
                    if self.callUp[i]:
                        self.direction = Direction.DOWN
                        return i
            else:
                for i in reversed(search_range[1]):  # pos -> 0
                    if self.callDown[i]:
                        return i
                for i in search_range[1]:  # 0 -> pos
                    if self.callUp[i]:
                        if self.position == i:
                            self.direction = Direction.UP
                        return i
                # Nach oben schauen und ggf. Richtung wechseln
                for i in search_range[0]:  # pos -> 14
                    if self.callUp[i]:
                        self.direction = Direction.UP
                        return i
                for i in reversed(search_range[0]):  # 14 -> pos
                    if self.callDown[i]:
                        self.direction = Direction.UP
                        return i

            if searched_one_way:
                break
            else:
                self.direction = Direction.UP if self.direction == Direction.DOWN else Direction.DOWN
                searched_one_way = True
        return None

    def operate(self):
        self.state = self.nextState

        if self.position == Conf.maxFloor - 1:
            self.direction = Direction.DOWN
        elif self.position == 0:
            self.direction = Direction.UP

        log: dict = dict()
        if (self.target == self.position and len(self.passengers)) or (  # Auftrag erfüllt
                self.isFloorRequested() and len(self.passengers) < self.capacity):
            #       Aufzug wird beim Vorbeifahren angefordert und hat noch Platz
            if self.state != ElevatorState.WAIT:
                self.nextState = ElevatorState.WAIT
                # Wenn gerade erst auf dem Stockwerk angekommen, einen Takt warten
            else:
                # Elevator is idle
                self.personsLeaving()
                self.personsEntering()
                target_floor = Conf.maxFloor - 1 if self.direction == Direction.UP else 0
                for p in self.passengers:
                    if self.direction == Direction.UP:
                        floor = p.schedule[0][1]
                        if floor <= target_floor:
                            self.nextState = ElevatorState.UP
                            target_floor = floor
                    else:
                        floor = p.schedule[0][1]
                        if floor >= target_floor:
                            self.nextState = ElevatorState.DOWN
                            target_floor = floor
                if self.nextState != ElevatorState.WAIT:
                    self.target = target_floor
        else:
            if len(self.passengers) == 0 and self.position == self.target:  # and self.state == ElevatorState.WAIT:
                target_floor = self.search_for_call()
                # TODO: Wenn auf selbe Etage, dann nimmt nicht mit, da Direction falsch
                if target_floor != None:
                    self.target = target_floor
                    if self.target == self.position:
                        self.nextState = ElevatorState.WAIT
                    else:
                        self.nextState = ElevatorState.UP if self.direction == Direction.UP else ElevatorState.DOWN
                else:
                    self.nextState = ElevatorState.WAIT
            if self.position != self.target:
                if len(self.passengers) > 0:
                    self.nextState = ElevatorState.UP if self.direction == Direction.UP else ElevatorState.DOWN
                self.position += self.state.value
                if self.position == Conf.maxFloor:
                    self.position = Conf.maxFloor - 1
                if self.position == -1:
                    self.position = 0

        log[f"({self.index}) position"] = self.position
        log[f"({self.index}) target"] = self.target
        log[f"({self.index}) state"] = self.state
        log[f"({self.index}) number of passangers"] = len(self.passengers)
        log[f"({self.index}) passangers"] = self.passengers
        Logger.add_data(log)

    def end_of_day(self):
        log: dict = dict()
        log[f"({self.index}) position"] = self.position
        log[f"({self.index}) number of passangers"] = len(self.passengers)
        log[f"({self.index}) passangers"] = self.passengers
        return log

    def __repr__(self):
        return f"index: {self.index}, position: {self.position}, target: {self.target}, passengers: {len(self.passengers)}, state: {self.state}, next state: {self.nextState}"
