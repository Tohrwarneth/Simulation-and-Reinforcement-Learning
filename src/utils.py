import argparse
import csv
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import pygame
from pathlib import Path
import os
import pandas as pd
from pandas.errors import EmptyDataError
from re_learner.IReinforcementDecider import IReinforcementDecider
from re_learner.reinforcment_decider import ReinforcementDecider


class Conf:
    """
    Config class
    """
    # Reinforcement Learner
    train: bool = False
    #
    # Elevator
    capacity: int = 5
    reinforcement_decider: IReinforcementDecider = ReinforcementDecider
    #
    # Resolution
    screenSize: tuple[float, float] = (1280, 720)
    screenOriginSize: tuple[float, float] = (1920, 1080)
    screenScale: tuple[float, float] = (1, 1)
    #
    # Building
    totalAmountPerson: int = 100
    maxFloor: int = 15
    #
    # Logs
    generatesPlots: bool = False
    showPlots: bool = False
    plotPath: str = "../paper/simulation/images"
    logPath: str = "logs"
    #
    # GUI
    fontSizeLarge: int = 30
    fontSizeSmall: int = 20
    fontLarge: pygame.font
    fontSmall: pygame.font

    @staticmethod
    def parse_args() -> tuple[bool, bool]:
        parser = argparse.ArgumentParser(prog='ElevatorSimulation',
                                         description='Simulates elevators of an office complex in a simple way')

        parser.add_argument('-p', '--plots', help="Generates plots", action='store_true')
        parser.add_argument('-sp', '--showPlots', help="Shows generated plots. (sets --plots true)",
                            action='store_true')
        parser.add_argument('-ui', '--ui', help="Shows user interface", action='store_true')
        parser.add_argument('-s', '--skip', help="Fast forward to hour x", type=int, nargs='?')
        parser.add_argument('-rl', '--reinforcementLearner',
                            help="Runs the simulation with reinforcement learned Decider", action='store_true')
        parser.add_argument('-t', '--train', help="Trains the reinforcement learner", action='store_true')
        parser.add_argument('-nl', '--noLogs', help="Doesn't generates log files", action='store_true')

        args = parser.parse_args()
        argument_dict: dict = vars(args)
        print(f"Dict format: {argument_dict}")

        Conf.generatesPlots = args.plots if args.plots or args.showPlots else False
        Conf.showPlots = args.showPlots
        Clock.skip = args.skip
        Conf.train = args.train
        Logger.noLogs = args.noLogs
        show_gui: bool = args.ui
        reinforcement_learning: bool = args.reinforcementLearner or args.train
        return show_gui, reinforcement_learning


class Clock:
    """
    Config, model and logic of the simulation clock/time
    """
    # Config
    peakTimes: list[tuple[int, int]] = [(8 * 60, 10), (12 * 60, 10), (17 * 60, 10)]
    breakDuration: int = 30
    #
    # Model
    endOfDay: bool = False
    running: bool = True
    pause: bool = False
    tact: int = 0
    speedScale: int = 1
    skip: int | None = None  # skip until
    #
    # Logic
    tactBuffer: int = 1
    timeInMin: float = 0
    speedPrePaused: int = 1

    @classmethod
    def reset(cls):
        # Model
        cls.endOfDay = False
        cls.running = True
        cls.pause = False
        cls.tact = 0
        # Logic
        cls.speedScale = 1
        cls.tactBuffer = 1
        cls.timeInMin = 0
        cls.speedPrePaused = 1

    @classmethod
    def add_time(cls, passed_time: float) -> None:
        """
        Adds delta time to tact
        :param passed_time: delta time
        :return: None
        """
        cls.timeInMin += passed_time
        cls.tactBuffer = int(cls.timeInMin) - cls.tact
        if cls.timeInMin > 24 * 60:
            cls.running = False
            cls.tactBuffer = 24 * 60 - cls.tact


class LogData:
    """
    Model of data as dictionary
    """
    tact: int = 0
    data: dict

    def __init__(self, tact: int):
        self.tact = tact
        self.data = {'tact': tact}

    def add_data(self, data: dict) -> None:
        """
        Merges data dictionary with logged data
        :param data: data dictionary
        :return: None
        """
        self.data = self.data | data


class Logger:
    """
    Save log files
    """
    csv: str | None
    eod_file: str  # End Of Day
    eod_session_file: str | None  # End Of Day for each session
    currentData: LogData | None = None
    allData: list[LogData]
    log_limits: int = 10
    dateTime: str
    noLogs: bool = False

    @classmethod
    def init(cls) -> None:
        """
        Initialize Logger
        :return:
        """
        if cls.noLogs:
            return
        now = datetime.now()  # current date and time
        cls.dateTime = now.strftime("%m.%d.%Y-%H.%M.%S")

        if Conf.train:
            cls.csv = None
        else:
            Path(f"{Conf.logPath}//{cls.dateTime}").mkdir(parents=True, exist_ok=True)
            cls.csv = f"{Conf.logPath}//{cls.dateTime}//run.csv"
            os.makedirs(os.path.dirname(cls.csv), exist_ok=True)

        if Conf.train:
            cls.eod_file = f"{Conf.logPath}/train_eod.csv"
            cls.eod_session_file = None
        else:
            cls.eod_file = f"{Conf.logPath}/eod.csv"
            cls.eod_session_file = f"{Conf.logPath}//{cls.dateTime}//eod.csv"

        logs = [name for name in os.listdir(Conf.logPath)
                if os.path.isdir(os.path.join(Conf.logPath, name))]
        while len(logs) > cls.log_limits:
            shutil.rmtree(f"{Conf.logPath}//{logs[0]}", ignore_errors=True)
            logs.pop(0)

        cls.allData = list()

    @classmethod
    def log(cls, plot_name: str = None) -> None:
        """
        Saves plot or current data
        :param plot_name: Name of the plot to save. None if data should be saved
        :return: None
        """
        if cls.noLogs:
            return
        if not plot_name:
            if Clock.endOfDay:
                cls.write_file(cls.eod_file, eod=True)
                if not Conf.train:
                    cls.write_file(cls.eod_session_file, eod=True)
            else:
                cls.write_file(cls.csv)
            cls.allData.append(cls.currentData)
            cls.currentData = None
        if plot_name:
            plt.savefig(f"{Conf.logPath}//{cls.dateTime}//{plot_name}", dpi=300)

    @classmethod
    def new_tact(cls) -> None:
        """
        Create new log data for the new tact
        :return: None
        """
        if cls.noLogs:
            return
        cls.currentData = LogData(Clock.tact)

    @classmethod
    def add_data(cls, data: dict) -> None:
        """
        Adds data dictionary to the current Log Data
        :param data: Data dictionary
        :return:
        """
        if cls.noLogs:
            return
        cls.currentData.add_data(data)

    @classmethod
    def write_file(cls, file: str, eod: bool = False) -> None:
        """
        Writes current log data to file
        :param file: file name
        :param eod: True if end of day log
        :return: None
        """
        if cls.noLogs:
            return
        assert file
        with open(file, "a", newline='') as csv_file:
            data_dict = cls.currentData.data
            w = csv.DictWriter(csv_file, data_dict.keys(), delimiter=',')
            if eod:
                try:
                    # Check if CSV is empty
                    df = pd.read_csv(file, delimiter=',', encoding="utf-8")
                except EmptyDataError:
                    w.writeheader()
            else:
                if len(cls.allData) == 0:
                    w.writeheader()
            w.writerow(data_dict)
