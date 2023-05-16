import pygame

from .GuiGlock import GuiGlock
from .IGuiObject import IGuiObject
from typing import List

from src_old.conf import Conf
from src_old.clock import Clock


class GuiManager:
    guiObjects: List[IGuiObject]
    clock: pygame.time.Clock
    running: bool
    step_gui: bool
    dt: float
    gameDisplay: pygame.Surface
    running: bool = True
    background_image: pygame.Surface

    def __init__(self, step_gui: bool):
        self.step_gui = step_gui
        self.guiObjects = list()
        screen_size = Conf.screen_size

        # pygame setup
        pygame.init()
        self.gameDisplay = pygame.display.set_mode(screen_size, pygame.RESIZABLE, vsync=1)


        self.clock = pygame.time.Clock()
        self.running = True
        self.dt = 0

        Conf.font = pygame.font.Font('freesansbold.ttf', 32)
        Conf.font_small = pygame.font.Font('freesansbold.ttf', 20)
        Conf.screen_scale = (screen_size[0] / Conf.screen_origin_size[0]
                             , screen_size[1] / Conf.screen_origin_size[1])

        self.update_screen_scale()

        self.gameDisplay.fill("pink")
        pygame.display.set_caption('Aufzugs Simulator')

        gui_glock = GuiGlock()
        self.add_gui_object(gui_glock)

    def initObjects(self):
        for gui in self.guiObjects:
            gui.init()

    def update(self, delta_time: float):
        for gui in self.guiObjects:
            gui.update(delta_time)

    def render(self):
        self.gameDisplay.blit(self.background_image, (0, 0))

        for gui in self.guiObjects:
            gui.render(self.gameDisplay)

    def update_screen_scale(self):
        self.background_image = pygame.image.load('images/FahrstuhlLayout.png')
        self.background_image.convert()
        self.background_image = pygame.transform.scale(
            self.background_image, Conf.screen_size)

        for gui in self.guiObjects:
            gui.update_screen_scale()

    def add_gui_object(self, object: IGuiObject) -> None:
        self.guiObjects.append(object)

    def frame(self) -> bool:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        self.handle_input()

        self.update(Clock.delta_time)

        self.render()
        # flip() the display to put your work on screen
        pygame.display.flip()

        # # limits FPS to 60
        # # dt is delta time in seconds since last frame, used for framerate-
        # # independent physics.
        # self.dt = self.clock.tick(60) / 1000

        while Clock.end_of_day:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        break
                    elif event.key == pygame.K_ESCAPE:
                        break
            self.running = False

        while self.step_gui:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        break
        return self.running

    def shutdown(self):
        pygame.quit()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                    Conf.speed_scale *= 2
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    Conf.speed_scale /= 2
                elif event.key == pygame.K_r:
                    Conf.speed_scale = 1
                elif event.key == pygame.K_HASH:
                    Conf.speed_scale = 128
            if event.type == pygame.VIDEORESIZE:
                width = event.w
                height = event.h
                Conf.screen_size = (width, height)
                Conf.screen_scale = (width / Conf.screen_origin_size[0]
                                     , height / Conf.screen_origin_size[1])
                self.update_screen_scale()
                self.gameDisplay = pygame.display.set_mode((width, height),
                                                           pygame.RESIZABLE)
                self.update_screen_scale()
