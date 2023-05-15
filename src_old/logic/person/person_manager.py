import random

import numpy
import numpy as np
import scipy
import simpy
from numpy.random import default_rng

from src_old.clock import Clock
from src_old.conf import Conf, LogData
from src_old.logic.person.Person import Person
import matplotlib.pyplot as plt

from src_old.logic.states import DayState
from src_old.ui.GuiFloor import GuiFloor


class PersonManager:
    persons: list[Person] = list()
    person_floor_up: list[simpy.Store]  # hoch
    person_floor_down: list[simpy.Store]  # runter
    env: simpy.Environment
    gui: GuiFloor
    leaved_person: int = 0
    rng: numpy.random.Generator

    def __init__(self, env: simpy.Environment):
        self.env = env
        self.rng = default_rng()
        total_person = Conf.total_amount_person
        max_floor = Conf.max_floor
        home_floors = scipy.stats.uniform.rvs(loc=1, scale=max_floor - 2, size=total_person)

        schedule_times: list[numpy.ndarray] = list()
        for j, (mean, std) in enumerate(Clock.get_peak()):
            times = self.get_times(mean, std)
            schedule_times.append(times)
            # self.draw_gamma(times)

        floors = [i for i in range(0, max_floor)]
        persons_for_plot = [i for i in range(0, max_floor)]
        for i in range(0, total_person):
            home_floor: int = int(home_floors[i])
            mensa_floor: int = 10 if home_floor > 7 else 5
            schedule: list[tuple[int, int]] = list()
            schedule.append((schedule_times[0][i], home_floor))
            schedule.append((schedule_times[1][i], mensa_floor))
            schedule.append((schedule_times[1][i] + Clock.pause_time, home_floor))
            schedule.append((schedule_times[2][i], 0))

            person = Person(i, schedule)
            self.persons.append(person)
            persons_for_plot[home_floor] += 1

        if Conf.show_plots:
            plt.plot(floors, persons_for_plot, linewidth=2, color='r')
            plt.xlabel('Etage')
            plt.ylabel('Personen')
            plt.show()
            plt.savefig(f'{Conf.plot_path}/Etagenverteilung.png')

        self.person_floor_up = [simpy.Store(env) for _ in range(0, Conf.max_floor)]
        self.person_floor_down = [simpy.Store(env) for _ in range(0, Conf.max_floor)]

    def init(self, gui: GuiFloor = None):
        self.gui = gui
        if gui:
            self.gui.person_floor = (self.person_floor_up, self.person_floor_down)

    @staticmethod
    def get_log_header() -> list[str]:
        # header: list[str] = [f"floor: {i}" for i in range(0, Conf.max_floor)]
        header: list[str] = [f"floor {i}" for i in range(0, Conf.max_floor)]
        return header

    def update(self) -> list:
        self.spawn()
        log_data: list = list()
        for i in range(Conf.max_floor):
            log_data.append((len(self.person_floor_up[i].items), len(self.person_floor_down[i].items)))
        if not self.gui == None:
            self.gui.person_floor = (self.person_floor_up, self.person_floor_down)

        return log_data

    def get_times(self, mean, std) -> numpy.ndarray:
        k = (mean / std) ** 2
        theta = std ** 2 / mean
        gamma: numpy.ndarray = self.rng.gamma(k, theta, size=Conf.total_amount_person)
        return gamma

    def spawn(self):
        for person in self.persons:
            time, floor = person.get_job()
            if self.env.now >= time and not (time, floor) == person.currentJob:
                current_floor = person.currentFloor
                if current_floor < floor:
                    self.person_floor_up[current_floor].put(person)
                elif current_floor > floor:
                    self.person_floor_down[current_floor].put(person)
                else:
                    print(current_floor, "=", floor, 4 * " ", person)
                person.currentJob = (time, floor)

    def get_person_on_floor(self, floor: int) -> tuple[list[Person], list[Person]]:
        return self.person_floor_up[floor].items, self.person_floor_down[floor].items

    def next_above(self, floor: int) -> int | None:
        for new_floor in range(floor, Conf.max_floor):
            if len(self.person_floor_down[new_floor].items) > 0 or len(self.person_floor_down[new_floor].items)>0:
                return new_floor
        return None

    def next_down(self, floor: int) -> int | None:
        for new_floor in reversed(range(floor)):
            if len(self.person_floor_down[new_floor].items) > 0 or len(self.person_floor_up[new_floor].items)>0:
                return new_floor
        return None

    @staticmethod
    def draw_gamma(gamma: np.ndarray):
        # Histogramm erstellen
        plt.hist(gamma, bins=1000, density=True, alpha=0.7)

        # Achsenbeschriftungen
        plt.xlabel('Werte')
        plt.ylabel('Dichte')

        # Titel des Plots
        plt.title('Gamma-Verteilung')

        # Diagramm anzeigen
        plt.show()
