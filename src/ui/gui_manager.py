import math
import time

import pygame
import simpy

from src.utils import Conf, Clock
from src.logic.elevator import Elevator
from src.ui.gui_clock import GuiClock
from src.ui.gui_floor import GuiFloor
from src.ui.elevator.gui_elevator import GuiElevator


class GuiManager:
    env: simpy.Environment
    elevatorList: list[Elevator]
    numOfElevators: int
    queUpward: list[list]
    queDownward: list[list]

    def __init__(self, elevator_list: list[Elevator],
                 que_upward: list[list], que_downward: list[list]):
        self.elevatorList = elevator_list
        self.numOfElevators = len(elevator_list)
        self.queUpward = que_upward
        self.queDownward = que_downward

        self.clock = GuiClock()
        self.floor = GuiFloor(call_up=self.queUpward, call_down=self.queDownward)
        self.elevatorGuiList = list()
        for i in range(self.numOfElevators):
            self.elevatorGuiList.append(GuiElevator(self.elevatorList[i]))

        pygame.init()
        self.screen = pygame.display.set_mode(Conf.screenSize, pygame.RESIZABLE)

        Conf.font = pygame.font.Font(pygame.font.get_default_font(), 20)  # 30
        Conf.fontSmall = pygame.font.Font(pygame.font.get_default_font(), 14)  # 20
        Conf.screenScale = (Conf.screenSize[0] / Conf.screenOriginSize[0]
                            , Conf.screenSize[1] / Conf.screenOriginSize[1])

        self.screen.fill("pink")
        pygame.display.set_caption('Aufzugs Simulator')

        self.update_screen_scale()

    def draw(self):
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                Clock.running = False
            if event.type == pygame.KEYDOWN:
                # space to pause
                if event.key == pygame.K_SPACE:
                    Clock.pause = not Clock.pause
                # press UP to double the speed
                if event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                    if Clock.skip:
                        Clock.skip = None
                        Clock.speedScale = 1.0
                    Clock.speedScale *= 2
                    Clock.skip = None
                # press Down to half the speed
                if event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    if Clock.skip:
                        Clock.skip = None
                        Clock.speedScale = 1.0
                    Clock.speedScale /= 2
            if event.type == pygame.VIDEORESIZE:
                width = event.w
                height = event.h
                Conf.screenSize = (width, height)
                Conf.screenScale = (width / Conf.screenOriginSize[0]
                                    , height / Conf.screenOriginSize[1])
                self.update_screen_scale()
                self.screen = pygame.display.set_mode((width, height),
                                                      pygame.RESIZABLE)
                self.update_screen_scale()

        self.screen.blit(self.background_image, (0, 0))

        self.clock.render(self.screen)

        self.floor.render(self.screen)

        for elevator in self.elevatorGuiList:
            elevator.render(self.screen)

        # flip() the display to put your work on screen
        pygame.display.flip()

    def update_screen_scale(self):
        self.background_image = pygame.image.load('images/FahrstuhlLayout.png')
        self.background_image.convert()
        self.background_image = pygame.transform.scale(
            self.background_image, Conf.screenSize)

        self.clock.update_screen_scale()
        self.floor.update_screen_scale()
