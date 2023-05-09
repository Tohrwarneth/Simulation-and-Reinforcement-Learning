import csv
from time import time

from src.logic.glock import Glock


class Conf:
    total_amount_person: int = 1000
    max_floor: int = 15
    show_plots: bool = False
    plot_path: str = "../paper/simulation/images"
    log_path: str = "logs"


class LogData:
    tact: int = 0
    person_per_floor: list[tuple]

    def __init__(self):
        self.tact = Glock.tact
        self.person_per_floor = [(i, 0) for i in range(0, Conf.max_floor)]

    def get_line(self) -> list[str]:
        line: list[str] = list()
        line += self.person_per_floor
        return line


class Log:
    csv: str
    currentData: LogData
    allData: list[LogData]

    @classmethod
    def init(cls):
        cls.csv = f"{Conf.log_path}/{int(time())}.csv"
        cls.currentData = LogData()

    @classmethod
    def log(cls):
        with open(cls.csv, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(cls.currentData.get_line())
            # writer.writerow(cls.currentData.get_line())
        cls.currentData = LogData()
