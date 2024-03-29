from logic.elevator import Elevator
import pygame
from ui.elevator.gui_passengers import GuiPassengers
from ui.elevator.gui_elevator_floor import GuiElevatorFloor
from ui.elevator.gui_jobs import GUIJobs
from ui.elevator.gui_status import GUIStatus


class GuiElevator:

    def __init__(self, elevator: Elevator):
        self.elevator = elevator

        self.statusGUI: GUIStatus = GUIStatus(elevator.index, elevator.position, elevator.state, elevator.direction)
        self.passengersGUI: GuiPassengers = GuiPassengers(elevator.index)
        self.floorGUI: GuiElevatorFloor = GuiElevatorFloor(elevator.index, elevator.position, elevator.callUp,
                                                           elevator.callDown)
        self.jobsGUI: list[GUIJobs] = list()
        for i in range(0, 5):
            self.jobsGUI.append(GUIJobs(elevator.index, i, elevator.passengers))

    def update(self) -> None:
        self.statusGUI.update(self.elevator.position, self.elevator.target, self.elevator.state, self.elevator.direction)
        self.passengersGUI.update(len(self.elevator.passengers))
        self.floorGUI.update(self.elevator.position)
        for job in self.jobsGUI:
            job.update(self.elevator.target)

    def render(self, game_display: pygame.Surface) -> None:
        self.update()

        self.statusGUI.render(game_display)
        self.passengersGUI.render(game_display)
        self.floorGUI.render(game_display)
        for job in self.jobsGUI:
            job.render(game_display)

    def update_screen_scale(self):
        self.statusGUI.update_screen_scale()
        self.passengersGUI.update_screen_scale()
        self.floorGUI.update_screen_scale()
        for job in self.jobsGUI:
            job.update_screen_scale()
