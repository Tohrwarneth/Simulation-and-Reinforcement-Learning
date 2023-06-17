import numpy as np
import scipy
from numpy.random import default_rng

from utils import Conf, Clock, Logger
from logic.person import Person


class PersonManager:
    """
    Manages the people in the building
    """
    # create persons
    scheduleTimes: np.ndarray
    homeFloors: list[float]
    #
    # managing
    persons: list[Person]
    atHome: list[Person]
    callUp: list[list[Person]]
    callDown: list[list[Person]]
    numberInMotion: list[tuple[int, int]]

    def __init__(self, call_up: list[list[Person]], call_down: list[list[Person]]):
        self.persons = list()
        self.atHome = list()
        self.callUp = call_up
        self.callDown = call_down
        self.numberInMotion = list()

        self.create_persons()

    def create_persons(self) -> None:
        """
        Generates people according to the config class
        :return: None
        """
        total_person = Conf.totalAmountPerson

        # Uniformly distributed home floors of people
        rng = default_rng()
        self.homeFloors = scipy.stats.uniform.rvs(loc=1, scale=Conf.maxFloor - 1, size=total_person)

        # Gamma distributed schedule times
        self.scheduleTimes: list[np.ndarray] = list()
        for j, (mean, std) in enumerate(Clock.peakTimes):
            k = (mean / std) ** 2
            theta = std ** 2 / mean
            times: np.ndarray = rng.gamma(k, theta, size=total_person)
            self.scheduleTimes.append(times)

        # Initialize Persons
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

    def manage(self) -> None:
        """
        Manage persons each tact
        :return: None
        """
        log: dict = dict()
        for person in self.persons:
            if person.waitingStartTime != None:
                continue
            elif not person.schedule:
                # if no schedule left, person has left the building
                person.position = 0
                self.persons.remove(person)
                self.atHome.append(person)
                continue
            else:
                # check if time to travel
                time_in_sec, target_floor = person.schedule[0]
                if Clock.tact >= time_in_sec:
                    if target_floor - person.position > 0:
                        self.callUp[person.position].append(person)
                    elif target_floor - person.position < 0:
                        self.callDown[person.position].append(person)
                    else:
                        person.schedule.pop(0)
                        continue
                    person.waitingStartTime = Clock.tact

        log['people in building'] = f"{self.get_remaining_people()}/{Conf.totalAmountPerson}"
        in_motion: int = self.get_people_in_motion()
        self.numberInMotion.append(in_motion)
        log['people in motion'] = f"{in_motion}/{Conf.totalAmountPerson}"
        log["call up"] = self.callUp
        log["call down"] = self.callDown
        Logger.add_data(log)

    def get_remaining_people(self) -> int:
        """
        Returns the number of remaining people in the building
        :return: number of remaining people
        """
        remaining_in_building = 0
        for person in self.persons + self.atHome:
            if person.schedule or person.position != 0:
                remaining_in_building += 1
        return remaining_in_building

    def get_people_in_motion(self) -> int:
        """
        Return the number of people who are moving between floors or waiting for an elevator
        :return: number of people moving
        """
        in_motion = 0
        for person in self.persons:
            if person.waitingStartTime:
                in_motion += 1
        return in_motion

    def end_of_day(self) -> dict:
        """
        Returns final log in dictionary
        :return: log dictionary
        """
        # for person in self.persons + self.atHome:
        #     if person.schedule or person.position != 0:
        #         print(person)
        log = {'remaining': f"{self.get_remaining_people()}/{Conf.totalAmountPerson}",
               'inMotion': f"{self.get_people_in_motion()}/{Conf.totalAmountPerson}"}
        return log
