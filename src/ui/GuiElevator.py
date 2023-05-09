from pygame import Rect

from .IGuiObject import IGuiObject
import pygame


class GUIStatus(IGuiObject):
    index: int

    def __init__(self, index: int):
        self.index = index

    def init(self, screen: pygame.surface) -> None:
        width = screen.get_width()
        height = screen.get_height()
        size = pygame.Vector2(width / 4.285, height / 12)

        left = (width / 4) * (self.index + 1)
        rect: Rect = Rect(left, height / 6, size[0], size[1])
        rectangle = pygame.draw.rect(screen, "purple", rect)

    def update(self, delta: int) -> None:
        pass

    def render(self) -> None:
        pass


class GUIJobs(IGuiObject):
    index: int
    elevatorIndex: int

    def __init__(self, elevatorIndex: int, index: int):
        self.elevatorIndex = elevatorIndex
        self.index = index

    def init(self, screen):
        width = screen.get_width()
        height = screen.get_height()
        size = pygame.Vector2(width / 15, height / 2)

        left = (width / 4) * (self.index + 1) + (width / 6)
        rect: Rect = Rect(left, height / 4, size[0], size[1])
        rectangle = pygame.draw.rect(screen, "red", rect)

    def update(self, delta: int) -> None:
        pass

    def render(self) -> None:
        pass


class GuiElevator(IGuiObject):
    capacity: int
    index: int
    rectangle: Rect
    screen: pygame.Surface
    statusGUI: GUIStatus
    jobsGUI: list[GUIJobs]

    def __init__(self, index: int):
        self.index = index
        self.statusGUI = GUIStatus(index)
        self.jobsGUI = list()
        for i in range(0, 5):
            self.jobsGUI.append(GUIJobs(index, i))

    def init(self, parentScreen):
        self.screen = parentScreen
        width = parentScreen.get_width()
        height = parentScreen.get_height()
        size = pygame.Vector2(width / 6, height / 2)
        # self.screen = pygame.display.set_mode((size[0], size[1]), pygame.SRCALPHA)

        left = (width / 4) * (self.index + 1)
        rect: Rect = Rect(left, height / 4, size[0], size[1])
        # rect: Rect = Rect(left, height / 4, 9*50, 16*50)
        rectangle = pygame.draw.rect(parentScreen, "#0047FF", rect)

        self.statusGUI.init(parentScreen)
        for job in self.jobsGUI:
            job.init(parentScreen)

        # self.screen.set_alpha(128)
        # self.screen.fill((255, 255, 255))
        # parentScreen.blit(self.screen, (0, 0))

    def update(self, delta: float) -> None:
        pass

    def render(self) -> None:
        pass

    def setCurrentFloor(self, floor: int):
        pass

    def setIndex(self, index: int):
        self.index = index


class Capacity(IGuiObject):
    index: int

    def __init__(self, index: int):
        self.index = index

    def init(self, screen):
        width = screen.get_width()
        height = screen.get_height()
        size = pygame.Vector2(width / 6, height / 2)

        left = (width / 4) * (self.index + 1)
        rect: Rect = Rect(left, height / 4, size[0], size[1])
        rectangle = pygame.draw.rect(screen, "#0047FF", rect)

    def update(self, delta: int) -> None:
        pass

    def render(self) -> None:
        pass
