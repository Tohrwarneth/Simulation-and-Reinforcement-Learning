import pygame

from .IGuiObject import IGuiObject
from typing import List


class GuiManager:
    guiObjects: List[IGuiObject]
    clock: pygame.time.Clock
    running: bool
    dt: float
    screen: pygame.Surface
    running: bool = True

    def init(self):
        self.guiObjects = list()

        # pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        self.clock = pygame.time.Clock()
        self.running = True
        self.dt = 0

        self.screen.fill("#F6E8C3")

        for gui in self.guiObjects:
            gui.init(self.screen)

    def update(self):
        for gui in self.guiObjects:
            gui.update(self.dt)

    def render(self):
        for gui in self.guiObjects:
            gui.render()

    def addGuiObject(self, object: IGuiObject) -> None:
        self.guiObjects.append(object)

    def frame(self) -> bool:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        self.update()

        self.render()
        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        self.dt = self.clock.tick(60) / 1000
        return self.running

    def shutdown(self):
        pygame.quit()
