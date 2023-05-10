import csv
from datetime import datetime

import pygame

from src.glock import Glock


class Conf:
    screen_size: tuple[float] = (1280, 720)
    total_amount_person: int = 1000
    max_floor: int = 15
    show_plots: bool = False
    plot_path: str = "../paper/simulation/images"
    log_path: str = "logs"
    font: pygame.font
    font_small: pygame.font
    skip: int = 0  # bis zur wievielten Stunde vorgespult werden soll
    speed_scale: int = 1


class LogData:
    tact: int = 0
    person_per_floor: list[tuple]

    def __init__(self):
        self.tact = Glock.tact
        self.person_per_floor = [0 for i in range(0, Conf.max_floor)]

    def get_line(self) -> list[str]:
        line: list[str] = list()
        line += [self.tact]
        line += self.person_per_floor
        return line

    @classmethod
    def get_header(cls) -> list[str]:
        line: list[str] = list()
        line += ["tact"]
        line += [f"floor {i}" for i in range(0, Conf.max_floor)]
        return line


from pathlib import Path
import os


class Log:
    csv: str
    currentData: LogData
    allData: list[LogData]
    log_limits: int = 10

    @classmethod
    def init(cls):
        Path(Conf.log_path).mkdir(parents=True, exist_ok=True)
        now = datetime.now()  # current date and time
        date_time = now.strftime("%d.%m.%Y-%H.%M.%S")
        cls.csv = f"{Conf.log_path}/{date_time}.csv"

        logs = [name for name in os.listdir(Conf.log_path)
                if os.path.isfile(os.path.join(Conf.log_path, name))]
        while len(logs) >= cls.log_limits:
            os.remove(f"{Conf.log_path}/{logs[0]}")
            logs.remove((logs[0]))

        cls.currentData = LogData()
        cls.allData = list()

        with open(cls.csv, "a", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(LogData.get_header())

    @classmethod
    def log(cls):
        if Glock.tact > cls.currentData.tact:
            with open(cls.csv, "a", newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(cls.currentData.get_line())
            cls.allData.append(cls.currentData)
            cls.currentData = LogData()
