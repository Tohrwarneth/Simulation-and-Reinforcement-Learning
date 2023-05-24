import pygame
from logic.person import Person
from utils import Conf


class GUIJobs:
    """
    GUI Representation of target floors of an elevator
    """
    elevatorIndex: int
    index: int
    passengers: list[Person]

    targetFloor: int
    position: tuple[float, float]

    def __init__(self, elevator_index: int, index: int, passengers: list[Person]):
        self.elevatorIndex = elevator_index
        self.index = index
        self.passengers = passengers

        self.targetFloor = 0
        self.position = (0, 0)

        self.update_screen_scale()

    def update(self, target_floor: int) -> None:
        """
        Update non referenced variable
        :param target_floor: current floor of the elevator
        :return: None
        """
        self.targetFloor = target_floor

    def render(self, game_display: pygame.Surface) -> None:
        """
        Rendering GUI
        :param game_display: master surface
        :return: None
        """
        for index, person in enumerate(self.passengers):
            floor: int = person.schedule[0][1]
            text_surface: pygame.Surface = Conf.fontSmall. \
                render(f"{floor + 1:01d}", True, "dark red" if floor == self.targetFloor else "black")
            text_rect: pygame.Rect = text_surface.get_rect()

            x, y = Conf.screenSize
            width: float = self.position[0]
            width_offset: float = x / 16
            height: float = self.position[1]
            height_offset: float = (y / 21 * index) + y / 20

            width = width_offset + width
            height = height_offset + height
            text_rect.center = (width, height)
            game_display.blit(text_surface, text_rect)

    def update_screen_scale(self) -> None:
        """
        Update position and font according to the new resolution
        :return: None
        """
        screen_size = Conf.screenSize
        offset: float = screen_size[0] / 2.858
        width: float = offset + (screen_size[0] / 3.55) * self.elevatorIndex
        height: float = (screen_size[1] / 2.62)
        self.position = (width, height)
