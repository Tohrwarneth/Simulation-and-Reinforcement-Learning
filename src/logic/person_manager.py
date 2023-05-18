import random

import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy.random import default_rng

from src.utils import Conf, Clock, Logger
from src.logic.person import Person


class PersonManager:
    scheduleTimes: np.ndarray
    homeFloors: list[float]

    def __init__(self, call_up: list[list[Person]], call_down: list[list[Person]]):
        self.callUp: list[list[Person]] = call_up
        self.callDown: list[list[Person]] = call_down

        self.persons: list = []
        self.atHome: list = []
        self.create_persons()

    def create_persons(self):
        '''
        Generates Persons based on peakTimes distribution
        '''
        rng = default_rng()
        total_person = Conf.totalAmountPerson
        max_floor = Conf.maxFloor
        self.homeFloors = scipy.stats.uniform.rvs(loc=1, scale=max_floor - 1, size=total_person)

        self.scheduleTimes: list[np.ndarray] = list()
        for j, (mean, std) in enumerate(Clock.peakTimes):
            k = (mean / std) ** 2
            theta = std ** 2 / mean
            times: np.ndarray = rng.gamma(k, theta, size=total_person)
            self.scheduleTimes.append(times)

        for i in range(0, total_person):
            home_floor: int = int(self.homeFloors[i])
            mensa_floor: int = 10 if home_floor > 7 else 5
            schedule: list[tuple[int, int]] = list()
            schedule.append((self.scheduleTimes[0][i], home_floor))
            if home_floor != 10 or home_floor != 5:
                schedule.append((self.scheduleTimes[1][i], mensa_floor))
                schedule.append((self.scheduleTimes[1][i] + Clock.breakDuration, home_floor))
            schedule.append((self.scheduleTimes[2][i], 0))

            person = Person(schedule=schedule)
            self.persons.append(person)

    def manage(self):
        '''
        adds persons to the queues at the time of their task + sets startWaitingTime
        removes persons if they have no tasks left
        '''
        log: dict = dict()
        for p in self.persons:
            # person has no tasks left on schedule -> can go home
            if not p.schedule:
                p.location = 0
                self.persons.remove(p)
                self.atHome.append(p)
                continue
            # Persons is already waiting in Que
            if p.startWaitingTime != None:
                continue
            # task
            time, floor = p.schedule[0]
            if Clock.tact >= time:
                if floor - p.location > 0:
                    self.callUp[p.location].append(p)
                    p.startWaitingTime = Clock.tact
                elif floor - p.location < 0:
                    # if (p not in self.QueDownward[p.location]):
                    self.callDown[p.location].append(p)
                    p.startWaitingTime = Clock.tact
                else:
                    # print(p)  # only prints in debug mode
                    # throw exception
                    p.schedule.pop()

        log['people in building'] = f"{self.get_remaining_people()}/{Conf.totalAmountPerson}"
        log['people in motion'] = f"{self.get_people_in_motion()}/{Conf.totalAmountPerson}"
        log["call up"] = self.callUp
        log["call down"] = self.callDown
        Logger.add_data(log)

    def get_remaining_people(self) -> int:
        remaining_in_building = 0
        for p in self.persons + self.atHome:
            if p.schedule or p.location != 0:
                remaining_in_building += 1
        return remaining_in_building

    def get_people_in_motion(self) -> int:
        in_motion = 0
        for p in self.persons:
            # if p.startWaitingTime != None:
            if p.startWaitingTime:
                in_motion += 1
        return in_motion

    def end_of_day(self) -> dict:
        log = {'remaining people in building': f"{self.get_remaining_people()}/{Conf.totalAmountPerson}"}
        return log
