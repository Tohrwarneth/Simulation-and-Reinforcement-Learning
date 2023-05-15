import abc
import pygame


class IGuiObject(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def init(self) -> None:
        pass

    @abc.abstractmethod
    def update(self, delta: float) -> None:
        pass

    @abc.abstractmethod
    def render(self, game_display: pygame.Surface) -> None:
        pass

    @abc.abstractmethod
    def update_screen_scale(self):
        pass
