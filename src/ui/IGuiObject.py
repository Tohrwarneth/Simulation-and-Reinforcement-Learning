import abc
import pygame


class IGuiObject(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def init(self, screen:pygame.Surface) -> None:
        pass

    @abc.abstractmethod
    def update(self, delta: float) -> None:
        pass

    @abc.abstractmethod
    def render(self) -> None:
        pass
