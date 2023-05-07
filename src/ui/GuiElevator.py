from pygame import Rect

from .IGuiObject import IGuiObject
import pygame


class GuiElevator(IGuiObject):
    capacity: int
    index: int
    rectangle: Rect
    screen: pygame.Surface

    def __init__(self, index: int):
        self.index = index

    def init(self, parentScreen):
        width = parentScreen.get_width()
        height = parentScreen.get_height()
        size = pygame.Vector2(width / 6, height / 2)
        self.screen = pygame.display.set_mode((size[0], size[1]), pygame.SRCALPHA)

        left = (width / 4) * (self.index + 1)
        rect: Rect = Rect(left, height / 4, size[0], size[1])
        rectangle = pygame.draw.rect(self.screen, "#0047FF", rect)

        self.screen.set_alpha(128)
        self.screen.fill((255, 255, 255))
        parentScreen.blit(self.screen, (0, 0))

    def update(self, delta: float) -> None:
        pass

    def render(self) -> None:
        pass

    def setCurrentFloor(self, floor: int):
        pass

    def setIndex(self, index: int):
        self.index = index


class Capacity(IGuiObject):

    def init(self, screen: pygame.surface) -> None:
        pass

    def update(self, delta: int) -> None:
        pass

    def render(self) -> None:
        pass
