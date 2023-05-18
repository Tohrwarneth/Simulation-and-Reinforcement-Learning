class Person:
    nextPersonIndex = 0

    def __init__(self, schedule, location=0):
        # private var
        self.index = Person.nextPersonIndex
        Person.nextPersonIndex += 1
        self.schedule = schedule  # List of taks [[10,3],[...]] => Task_0: 10:00 on Floor 3
        self.location = location  # current Floor the Person
        self.startWaitingTime: int | None = None  # set when Queued Up, None if Person is not waiting

    def __repr__(self):
        return f"{self.index}., location: {self.location}, schedule: {self.schedule}"
