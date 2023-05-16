import pygame

from src.logic.person import Person
from src.utils import Conf


class GUIJobs:
    elevatorIndex: int
    index: int

    def __init__(self, elevator_index: int, index: int, passengers: list[Person]):
        self.elevatorIndex = elevator_index
        self.index = index
        self.jobs = passengers

        self.target_floor: int = 0
        self.position: tuple[float, float] = (0, 0)

        self.update_screen_scale()

    def update(self, target_floor: int) -> None:
        self.target_floor = target_floor

    def render(self, game_display: pygame.Surface) -> None:
        sw, sh = Conf.screenScale
        for index, person in enumerate(self.jobs):
            floor: int = person.schedule[0][1]
            text_surface: pygame.Surface = Conf.fontSmall. \
                render(f"{floor:01d}", True, "dark red" if floor == self.target_floor else "black")
            text_rect: pygame.Rect = text_surface.get_rect()

            text_surface = \
                pygame.transform.scale(
                    text_surface,
                    (text_surface.get_width() * sw, text_surface.get_height() * sh))

            x, y = Conf.screenSize
            width = self.position[0]
            width_offset = x / 16
            height = self.position[1]
            height_offset = y / 21 * index + y / 20

            text_rect.center = (width_offset + width, height_offset + height)
            game_display.blit(text_surface, text_rect)

    def update_screen_scale(self):
        screen_size = Conf.screenSize
        offset = screen_size[0] / 2.858
        width = offset + (screen_size[0] / 3.55) * self.index
        height = (screen_size[1] / 2.62)
        self.position = (width, height)
