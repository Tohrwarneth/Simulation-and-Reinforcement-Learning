import csv
import shutil
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
    fontSizeLarge: int = 30
    fontSizeSmall: int = 20
    fontLarge: pygame.font
    fontSmall: pygame.font


class Clock:
    end_of_day: bool = False
    deltaTime: float = 0.0
    tact: int = 0
    tactBuffer: int = 1
    timeInMin: float = 0
    skip: int | None = None  # bis zur wievielten Stunde vorgespult werden soll
    peakTimes = [(8 * 60, 1), (13 * 60, 1), (17 * 60, 1)]
    breakDuration = 30
    speedScale: int = 1
    speedPrePaused: int = 1
    running: bool = True
    pause: bool = False

    @classmethod
    def add_time(cls, passed_time: float):
        cls.timeInMin += passed_time
        cls.tactBuffer = int(cls.timeInMin) - cls.tact
        if cls.timeInMin > 24 * 60:
            cls.running = False


class LogData:
    tact: int = 0
    data: dict

    def __init__(self, tact: int):
        self.tact = tact
        self.data = {'tact': tact}

    def add_data(self, data: dict):
        self.data = self.data | data


class Logger:
    csv: str
    eod_file: str  # End Of Day
    currentData: LogData | None = None
    allData: list[LogData]
    log_limits: int = 5

    @classmethod
    def init(cls):
        now = datetime.now()  # current date and time
        date_time = now.strftime("%d.%m.%Y-%H.%M.%S")
        Path(f"{Conf.logPath}/{date_time}").mkdir(parents=True, exist_ok=True)
        cls.csv = f"{Conf.logPath}/{date_time}/run.csv"
        cls.eod_file = f"{Conf.logPath}/eod.csv"

        logs = [name for name in os.listdir(Conf.logPath)
                if os.path.isdir(os.path.join(Conf.logPath, name))]
        while len(logs) > cls.log_limits:
            shutil.rmtree(f"{Conf.logPath}/{logs[0]}", ignore_errors=True)
            logs.pop(0)

        cls.allData = list()

    @classmethod
    def log(cls):
        if Clock.end_of_day:
            cls.write_file(cls.eod_file, eod=True)
        else:
            cls.write_file(cls.csv)
        cls.allData.append(cls.currentData)
        cls.currentData = None

    @classmethod
    def new_tact(cls):
        cls.currentData = LogData(Clock.tact)

    @classmethod
    def add_data(cls, data):
        cls.currentData.add_data(data)

    @classmethod
    def write_file(cls, file: str, eod: bool = False):
        with open(file, "a", newline='') as csv_file:
            data_dict = cls.currentData.data
            w = csv.DictWriter(csv_file, data_dict.keys(), delimiter=',')
            if len(cls.allData) == 0:
                w.writeheader()
            w.writerow(data_dict)
