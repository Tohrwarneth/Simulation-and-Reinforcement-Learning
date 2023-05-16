import csv
from datetime import datetime

import pygame
from pathlib import Path
import os


class Conf:
    screenSize: tuple[float, float] = (1280, 720)
    screenOriginSize: tuple[float, float] = (1920, 1080)
    screenScale: tuple[float, float] = (1, 1)
    totalAmountPerson: int = 100
    maxFloor: int = 15
    showPlots: bool = False
    plotPath: str = "../paper/simulation/images"
    logPath: str = "logs"
    font: pygame.font
    fontSmall: pygame.font
    deltaTime: float = 0.1


class Clock:
    tact: int = 0
    tactBuffer: int = 1
    timeInMin: float
    skip: int = 0  # bis zur wievielten Stunde vorgespult werden soll
    peakTimes = [(8 * 60, 1), (13 * 60, 1), (17 * 60, 1)]
    breakDuration = 30
    speedScale: int = 1
    running: bool = True
    pause: bool = False

    @classmethod
    def add_time(cls, passed_time: float):
        cls.timeInMin += passed_time
        if cls.timeInMin == 24 * 60:
            cls.running = False


class LogData:
    tact: int = 0
    header: list[str] = list()
    data: list

    def __init__(self, tact: int):
        self.tact = tact
        self.person_per_floor = [(0, 0) for i in range(0, Conf.maxFloor)]

    def get_line(self) -> list[str]:
        line: list = [self.tact]
        line += self.data
        return line

    @classmethod
    def get_header(cls) -> list[str]:
        line = ["tact"]
        line += cls.header
        return line

    def add_data(self, data):
        self.data = data

    @classmethod
    def add_header(cls, header):
        cls.header = header


class Log:
    csv: str
    currentData: LogData
    allData: list[LogData]
    log_limits: int = 5

    @classmethod
    def init(cls):
        Path(Conf.logPath).mkdir(parents=True, exist_ok=True)
        now = datetime.now()  # current date and time
        date_time = now.strftime("%d.%m.%Y-%H.%M.%S")
        cls.csv = f"{Conf.logPath}/{date_time}.csv"

        logs = [name for name in os.listdir(Conf.logPath)
                if os.path.isfile(os.path.join(Conf.logPath, name))]
        while len(logs) >= cls.log_limits:
            os.remove(f"{Conf.logPath}/{logs[0]}")
            logs.remove((logs[0]))

        cls.allData = list()

        with open(cls.csv, "a", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(LogData.get_header())

    @classmethod
    def log(cls, data: LogData):
        with open(cls.csv, "a", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(data.get_line())
        cls.allData.append(data)
