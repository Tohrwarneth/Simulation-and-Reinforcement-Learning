class Person:
    new_id = 0

    def __init__(self, schedule, location=0):
        Person.new_id += 1
        # private var
        self.id = Person.new_id
        self.schedule = schedule  # List of taks [[10,3],[...]] => Task_0: 10:00 on Floor 3
        self.location = location  # current Floor the Person
        self.startWaitingTime = None  # set when Queued Up, None if Person is not waiting