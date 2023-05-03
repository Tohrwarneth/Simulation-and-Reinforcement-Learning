from IGuiObject import IGuiObject
import pygame


class GuiElevator(IGuiObject):
    capacity: int

    def init(self, screen:pygame.Surface) -> None:
        pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
        pygame.draw.rect(screen, "#0047FF", pos, 40)

    def update(self, delta: float) -> None:
        pass

    def render(self) -> None:
        pass

    def setCurrentFloor(self, floor:int):
        pass

class Capacity(IGuiObject):

    def init(self, screen:pygame.surface) -> None:
        pass

    def update(self, delta: int) -> None:
        pass

    def render(self) -> None:
        pass
