from pygame import Rect

from src.logic.states import ElevatorState
from src.ui.IGuiObject import IGuiObject
import pygame

from src.ui.elevator.GuiCapacity import GuiCapacity
from src.ui.elevator.GuiFloor import GuiFloor
from src.ui.elevator.GuiJobs import GUIJobs
from src.ui.elevator.GuiStatus import GUIStatus


class GuiElevator(IGuiObject):
    capacity: int
    index: int
    state: ElevatorState
    currentFloor: int = 0
    target_floor: int = 0
    person_floor: tuple[int, int]
    jobs: list[int]
    rectangle: Rect
    screen: pygame.Surface
    statusGUI: GUIStatus
    jobsGUI: list[GUIJobs]
    capacityGUI: GuiCapacity
    floorGUI: GuiFloor

    def __init__(self, index: int):
        self.index = index
        self.jobs = list()
        self.statusGUI = GUIStatus(index, self.currentFloor)
        self.capacityGUI = GuiCapacity(index)
        self.floorGUI = GuiFloor(index)
        self.jobsGUI = list()
        self.person_floor = (0, 0)
        for i in range(0, 5):
            self.jobsGUI.append(GUIJobs(index, i))

    def init(self):
        self.statusGUI.init()
        self.capacityGUI.init()
        self.floorGUI.init()
        for job in self.jobsGUI:
            job.init()

    def update(self, delta_time: float) -> None:
        self.statusGUI.update(self.currentFloor, self.target_floor, self.state)
        self.capacityGUI.update(delta_time)
        self.floorGUI.update(self.person_floor)
        for job in self.jobsGUI:
            job.update(self.jobs, self.target_floor)

    def render(self, game_display: pygame.Surface) -> None:
        self.statusGUI.render(game_display)
        self.capacityGUI.render(game_display)
        self.floorGUI.render(game_display)
        for job in self.jobsGUI:
            job.render(game_display)

    def setCurrentFloor(self, floor: int):
        self.currentFloor = floor

    def setTargetFloor(self, floor: int):
        self.target_floor = floor

    def setIndex(self, index: int):
        self.index = index

    def setJobs(self, jobs: list[int]):
        self.jobs = jobs

    def update_screen_scale(self):
        self.statusGUI.update_screen_scale()
        self.capacityGUI.update_screen_scale()
        self.floorGUI.update_screen_scale()
        for job in self.jobsGUI:
            job.update_screen_scale()
