import random

import numpy
import numpy as np
import scipy
import simpy
from numpy.random import default_rng

from src.clock import Clock
from src.conf import Conf, LogData
from src.logic.person.Person import Person
import matplotlib.pyplot as plt

from src.logic.states import DayState
from src.ui.GuiFloor import GuiFloor


class PersonManager:
    to_building: list[Person] = list()
    to_work0: list[Person] = list()
    to_lunch: list[Person] = list()
    to_work1: list[Person] = list()
    to_home: list[Person] = list()
    person_floor_up: list[simpy.Store]  # hoch
    person_floor_down: list[simpy.Store]  # runter
    gui: GuiFloor
    leaved_person: int = 0
    gamma: list[int] = list()
    rng: numpy.random.Generator

    def __init__(self, env: simpy.Environment):
        total_person = Conf.total_amount_person
        max_floor = Conf.max_floor
        target_floors = scipy.stats.uniform.rvs(loc=1, scale=max_floor - 2, size=total_person)

        floors = [i for i in range(0, max_floor)]
        persons = [i for i in range(0, max_floor)]
        for i in range(0, total_person):
            target_floor: int = int(target_floors[i])
            person = Person(i, target_floor)
            self.to_building.append(person)
            persons[target_floor] += 1

        if Conf.show_plots:
            plt.plot(floors, persons, linewidth=2, color='r')
            plt.xlabel('Etage')
            plt.ylabel('Personen')
            plt.show()
            plt.savefig(f'{Conf.plot_path}/Etagenverteilung.png')

        self.person_floor_up = [simpy.Store(env) for i in range(0, Conf.max_floor)]
        self.person_floor_down = [simpy.Store(env) for i in range(0, Conf.max_floor)]

    def init(self, gui: GuiFloor = None):
        self.gui = gui

    @staticmethod
    def get_log_header() -> list[str]:
        # header: list[str] = [f"floor: {i}" for i in range(0, Conf.max_floor)]
        header: list[str] = ["to building"]
        return header

    def update(self, tact: int) -> list:
        self.spawn()
        log_data: list = list()
        # log_data += self.person_floor
        if not self.gui == None:
            self.gui.person_floor = (self.person_floor_up, self.person_floor_down)

        tmp = ""
        for person in self.to_building:
            tmp += f"{person}\n"
        tmp = f"\"{tmp}\""
        log_data += [tmp]
        return log_data

    def enter_building(self):
        person: Person = self.to_building.pop(0)
        self.to_work0.append(person)
        self.person_floor_up[0].put(person)

    def spawn(self):
        rng = default_rng()
        k = (Clock.get_peak() / 2.0) ** 2
        if not k == 0:
            theta = Clock.get_peak() / k
            gamma = rng.gamma(k, theta)
        else:
            gamma = 0
        self.gamma.append(gamma)
        morning, lunch, evening = Clock.peak_times
        if Clock.tact > morning[2]:
            gamma += len(self.to_building)
        elif Clock.tact > lunch[2]:
            gamma += len(self.to_lunch)
        elif Clock.tact > lunch[2] + 30:
            gamma += len(self.to_work1)
        elif Clock.tact > evening[2]:
            gamma += len(self.to_home)

        for i in range(0, int(gamma)):
            if len(self.to_building) > 0:
                self.enter_building()
            if len(self.to_lunch) > 0:
                self.enter_floor(DayState.PRE_LUNCH)
            if len(self.to_work1) > 0:
                self.enter_floor(DayState.POST_LUNCH)
            if len(self.to_home) > 0:
                self.enter_floor(DayState.EVENING)

    def enter_floor(self, state: DayState):
        person: Person
        if state == DayState.PRE_LUNCH:
            person: Person = self.to_lunch.pop(0)
        elif state == DayState.POST_LUNCH:
            person: Person = self.to_work1.pop(0)
        elif state == DayState.EVENING:
            person: Person = self.to_home.pop(0)
        else:
            return

        if person.targetFloor > person.homeFloor:
            self.person_floor_up[person.homeFloor].put(person)
        else:
            self.person_floor_down[person.homeFloor].put(person)

    def reached_target(self, person: Person):
        person.dayState += 1
        state: DayState = person.dayState
        if state == DayState.PRE_LUNCH:
            person.targetFloor = 10 if person.targetFloor > 7 else 5
            self.to_work0.remove(person)
            self.to_lunch.append(person)
        elif state == DayState.POST_LUNCH:
            person.targetFloor = person.homeFloor
            self.to_lunch.remove(person)
            self.to_work1.append(person)
        elif state == DayState.EVENING:
            person.targetFloor = 0
            self.to_work1.remove(person)
            self.to_home.append(person)
        else:
            self.to_home.remove(person)
            self.leaved_person += 1

    def get_person_on_floor(self, floor: int) -> tuple[list[Person], list[Person]]:
        return self.person_floor_up[floor].items, self.person_floor_down[floor].items

    def next_above(self, floor: int) -> int | None:
        for new_floor in range(floor, Conf.max_floor):
            if len(self.person_floor_down[new_floor].items) > 0:
                return new_floor
        return None

    def next_down(self, floor: int) -> int | None:
        for new_floor in range(0, floor).__reversed__():
            if len(self.person_floor_down[new_floor].items) > 0:
                return new_floor
        return None
