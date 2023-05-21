class Person:
    """
    Data model of a single person
    """
    nextPersonIndex: int = 0
    index: int
    schedule: list[tuple[float, int]]  # schedule, (when, where) the person start to go to
    #                                  # (List of tasks). Task removed if finished
    position: int
    waitingStartTime: int | None  # time, when started waiting for an elevator
    #
    homeFloor: int  # for debug

    def __init__(self, schedule):
        self.index = Person.nextPersonIndex
        Person.nextPersonIndex += 1
        self.schedule = schedule
        self.homeFloor = schedule[0][1]
        self.position = 0
        self.waitingStartTime = None

    def __repr__(self):
        return f"{self.index}.\t home: {self.homeFloor},\t position: {self.position}," \
               f"\t waiting since: {self.waitingStartTime},\t schedule: {self.schedule}"
