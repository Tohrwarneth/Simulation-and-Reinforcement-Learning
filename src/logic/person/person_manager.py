import random

import numpy as np
import scipy

from src.conf import Conf
from src.logic.person.Person import Person
import matplotlib.pyplot as plt

from src.ui.GuiFloor import GuiFloor


class PersonManager:
    to_building: list[Person] = list()
    to_work0: list[Person] = list()
    to_lunch: list[Person] = list()
    to_work1: list[Person] = list()
    to_home: list[Person] = list()
    person_floor: list[tuple[int, int]]
    gui: GuiFloor

    def __init__(self):
        total_person = Conf.total_amount_person
        max_floor = Conf.max_floor
        target_floors = scipy.stats.uniform.rvs(loc=1, scale=max_floor - 2, size=total_person)

        floors = [i for i in range(0, max_floor)]
        persons = [i for i in range(0, max_floor)]
        for i in range(0, total_person):
            target_floor: int = int(target_floors[i])
            person = Person(target_floor)
            self.to_building.append(person)
            persons[target_floor] += 1

        if Conf.show_plots:
            plt.plot(floors, persons, linewidth=2, color='r')
            plt.xlabel('Etage')
            plt.ylabel('Personen')
            plt.show()
            plt.savefig(f'{Conf.plot_path}/Etagenverteilung.png')

        self.person_floor = [(0, 0) for i in range(0, Conf.max_floor)]
        # TODO: Remove random
        # random.seed(a=None, version=2)
        # for i in range(0, Conf.max_floor):
        #     self.person_floor.append((random.randint(0, 100), random.randint(1, 100)))
        # self.person_floor[10] = (1, 0)
        # self.person_floor[5] = (0, 2)

    def init(self, gui: GuiFloor = None):
        self.gui = gui

    def update(self):
        if not self.gui == None:
            self.gui.person_floor = self.person_floor
