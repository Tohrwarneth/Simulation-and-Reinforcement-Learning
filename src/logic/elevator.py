from src.utils import Conf, Clock
from src.enums import ElevatorState
from src.logic.person import Person


class Elevator:
    nextElevatorIndex: int = 0
    index: int
    passengers: list[Person]
    state: ElevatorState = ElevatorState.WAIT
    position: int
    direction: int
    target: int

    # TODO: zu job Ziel fahren und wenn auf dem Weg jmd. ist und Platz ist, einpacken.
    #  Wenn nix zu tun, schau zuerst in deine ursprüngliche Richtung weiter, wenn da nix ist, dann unter dir.

    def __init__(self, QueUpward, QueDownward, capacity, speed=1, waitingTime=1, startPos=0):
        self.index = self.nextElevatorIndex
        Elevator.nextElevatorIndex += 1
        self.target = 0
        # private var
        self.call_down = QueDownward
        self.call_up = QueUpward
        self.waitingList: list[int] = []
        self.passengers = []
        self.capacity = capacity
        self.speed = speed
        self.direction = 0
        self.position = startPos
        self.waitingTime = waitingTime

    def isPassengerLeaving(self):
        '''
        checks if a passenger wants to leave on the current floor
        Returns:

        '''
        result = False
        for p in self.passengers:
            if p.schedule[0][1] == self.position:
                result = True

        return result

    def personsLeaving(self):
        '''
        Iterates over passengers
        sets there new location
        removes task from schedule
        resets there startwaitingTime = None
        accumulates the waitingTime in waitingList

        '''
        for p in self.passengers:
            if (p.schedule[0][1] == self.position):  # first task, location TODO use dict or docu better
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
        if self.direction == 0:
            while len(self.passengers) < self.capacity and (
                    self.call_up[self.position] or self.call_down[self.position]):
                if self.call_up[self.position]:
                    p = self.call_up[self.position].pop(0)
                    self.passengers.append(p)

                if self.call_down[self.position]:
                    p = self.call_down[self.position].pop(0)
                    self.passengers.append(p)
        elif self.direction == 1:
            while len(self.passengers) < self.capacity and self.call_up[self.position]:
                p = self.call_up[self.position].pop(0)
                self.passengers.append(p)
            while len(self.passengers) < self.capacity and self.call_down[self.position]:
                p = self.call_down[self.position].pop(0)
                self.passengers.append(p)
        else:
            while len(self.passengers) < self.capacity and self.call_down[self.position]:
                p = self.call_down[self.position].pop(0)
                self.passengers.append(p)
            while len(self.passengers) < self.capacity and self.call_up[self.position]:
                p = self.call_up[self.position].pop(0)
                self.passengers.append(p)

    def isFloorRequested(self):
        '''
        checks if the Floor at the current position is requested
        '''
        return (self.call_down[self.position]
                or self.call_up[self.position]
                or any(p.schedule[0][1] == self.position for p in self.passengers))

    def isFloorRequestedUpwards(self):
        '''
        Returns: is a person waiting on the current floor to go upwards
        '''
        return (self.call_up[self.position]
                or any(p.schedule[0][1] == self.position for p in self.passengers))

    def isFloorRequestedDownwards(self):
        '''
        Returns: is a person waiting on the current floor to go downwards
        '''
        return (self.call_down[self.position]
                or any(p.schedule[0][1] == self.position for p in self.passengers))

    def isFloorAboveRequested(self):
        '''
        checks if a Floor above current position is requested
        '''
        result = False
        if any(p.schedule[0][1] > self.position for p in self.passengers):
            result = True

        for i in range(self.position + 1, len(self.call_up)):
            if (self.call_up[i] or self.call_down[i]):
                result = True

        return result

    # TODO currently not used
    def isFloorAboveRequestedUp(self):
        '''
        checks if a Floor above current position is requested
        '''
        result = False
        if any(p.schedule[0][1] > self.position for p in self.passengers):
            result = True

        for i in range(self.position + 1, len(self.call_up)):
            if self.call_up[i]:
                result = True

        return result

    def isFloorBelowRequested(self):
        '''
        checks if a Floor below current position is Requested
        '''
        result = False
        if any(p.schedule[0][1] < self.position for p in self.passengers):
            result = True

        for i in range(self.position):  # TODO numOfFloors as attribute
            if self.call_down[i] or self.call_up[i]:
                result = True

        return result

    # TODO currently not used
    def isFloorBelowRequestedDown(self):
        '''
        checks if a Floor below current position is Requested
        '''
        result = False
        if any(p.schedule[0][1] < self.position for p in self.passengers):
            result = True

        for i in range(self.position):  # TODO numOfFloors as attribute
            if self.call_down[i]:
                result = True

        return result

    def operate(self):
        # Elevator going Up
        if self.isFloorAboveRequested():
            self.direction = 1

            # yield self.env.timeout(1 / self.speed)  # speed = Floors/ min
            self.position += self.direction

            if self.isFloorRequestedDownwards():
                # Elevator Waiting
                self.direction = 0
                # yield self.env.timeout(self.waitingTime)
                self.direction = 1

                self.personsLeaving()
                self.personsEntering()

            if not self.isFloorAboveRequested():
                # TODO check direction of Request + ERROR sometimes elevator goes up to 15
                self.direction = 0

        # Elevator going Down
        elif self.isFloorBelowRequested():
            self.direction = -1
            # Elevator going Down
            # yield self.env.timeout(1 / self.speed)  # speed = Floors/ min
            self.position += self.direction
            if self.isFloorRequestedUpwards():
                # Elevator Waiting
                self.direction = 0
                # yield self.env.timeout(self.waitingTime)
                self.direction = -1

                # removes and adds passengers
                self.personsLeaving()
                self.personsEntering()
            if not self.isFloorBelowRequested():
                self.direction = 0

        # Elevator is idle
        elif self.direction == 0 and self.isFloorRequested():
            self.personsEntering()

        self.target = self.position + self.direction
