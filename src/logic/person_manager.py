import random

import numpy as np

from src import Conf
from src.logic.person import Person


class PersonManager:

    def __init__(self, env, QueUpward, QueDownward):
        self.env = env
        self.QueUpward = QueUpward
        self.QueDownward = QueDownward

        self.persons = []

        self.spawn()
        self.env.process(self.coordinate())

    def spawn(self):
        '''
        generates Persons based on peakTimes distribution
        '''

        for _ in range(Conf.total_amount_person):
            schedule = []
            prev_floor = None

            for i in range(len(Conf.peakTimes)):
                mean, std = Conf.peakTimes[i]
                current_floor = random.randint(1, Conf.max_floor - 1)
                while current_floor == prev_floor:
                    current_floor = random.randint(1, Conf.max_floor - 1)

                k, theta = self.get_gamma(mean, std)
                schedule.append((np.random.gamma(k, theta), current_floor))
                prev_floor = current_floor

            self.persons.append(Person(schedule=schedule))

    def coordinate(self):
        '''
        adds persons to the queues at the time of their task + sets startWaitingTime
        removes persons if they have no tasks left
        '''
        while True:
            for p in self.persons:

                # person has no tasks left on schedule -> can go home
                if (not p.schedule):
                    self.persons.remove(p)
                    continue
                # Persons is already waiting in Que
                if (p.startWaitingTime != None):
                    continue
                # task
                time, floor = p.schedule[0]
                if (self.env.now >= time):
                    if (floor - p.location > 0):

                        self.QueUpward[p.location].append(p)
                        p.startWaitingTime = self.env.now
                        # print(f"Time: {self.env.now} Person {p.id} was added to the UpwardQue on floor {p.location}")  # only prints in debug mode
                    elif (floor - p.location < 0):
                        # if (p not in self.QueDownward[p.location]):
                        self.QueDownward[p.location].append(p)
                        p.startWaitingTime = self.env.now
                        # print(f"Time: {self.env.now} Person {p.id} was added to the DownwardQue on floor {p.location}")  # only prints in debug mode

                    else:
                        # print("ERROR: Same location as task location")  # only prints in debug mode
                        # throw exception
                        p.schedule.pop()
            yield self.env.timeout(Conf.deltaTime)

    def get_gamma(self, mean, std):
        '''
        Args:
            mean: expected Value
            std: standard deviation

        Returns: k,theta as parameter of gamma
        '''
        k = (mean / std) ** 2
        theta = std ** 2 / mean
        return k, theta
