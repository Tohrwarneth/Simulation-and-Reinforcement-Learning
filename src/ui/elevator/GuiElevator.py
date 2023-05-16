from pygame import Rect

from src.enums import ElevatorState
from src.logic.elevator import Elevator
import pygame

from src.ui.elevator.gui_passengers import GuiPassengers
from src.ui.elevator.gui_elevator_floor import GuiElevatorFloor
from src.ui.elevator.gui_jobs import GUIJobs
from src.ui.elevator.gui_status import GUIStatus


# TODO: Rework
class GuiElevator:
    passengers: int
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
    passengersGUI: GuiPassengers
    floorGUI: GuiElevatorFloor

    def __init__(self, elevator: Elevator):
        self.elevator = elevator

        self.statusGUI = GUIStatus(elevator.index, elevator.position, elevator.state)
        self.passengersGUI = GuiPassengers(elevator.index)
        self.floorGUI = GuiElevatorFloor(elevator.index, elevator.call_up, elevator.call_down)
        self.jobsGUI = list()
        for i in range(0, 5):
            self.jobsGUI.append(GUIJobs(elevator.index, i, elevator.passengers))

    def update(self, delta_time: float) -> None:
        self.statusGUI.update(self.currentFloor, self.target_floor, self.state)
        self.passengersGUI.update(len(self.elevator.passengers))
        # Floor GUI braucht kein Update, da er eine Ref. auf die Listen der Etagen-Calls hält
        for job in self.jobsGUI:
            job.update(self.target_floor)

    def render(self, game_display: pygame.Surface) -> None:
        self.statusGUI.render(game_display)
        self.passengersGUI.render(game_display)
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
        self.passengersGUI.update_screen_scale()
        self.floorGUI.update_screen_scale()
        for job in self.jobsGUI:
            job.update_screen_scale()
