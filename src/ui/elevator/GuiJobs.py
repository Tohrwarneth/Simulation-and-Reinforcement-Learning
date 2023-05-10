import pygame

from src.conf import Conf


class GUIJobs:
    index: int
    elevatorIndex: int
    target_floor: int
    jobs: list[int]
    position: tuple[float, float]

    def __init__(self, elevator_index: int, index: int):
        self.elevatorIndex = elevator_index
        self.index = index
        self.jobs = list()
        self.update_screen_scale()

    def init(self):
        pass

    def update_screen_scale(self):
        screen_size = Conf.screen_size
        offset = screen_size[0] / 2.858
        width = offset + (screen_size[0] / 3.55) * self.index
        height = (screen_size[1] / 2.62)
        self.position = (width, height)

    def update(self, jobs: list[int], target_floor: int) -> None:
        self.jobs = jobs
        self.target_floor = target_floor

    def render(self, game_display: pygame.Surface) -> None:
        sw, sh = Conf.screen_scale
        for index, floor in enumerate(self.jobs):
            text_surface: pygame.Surface = Conf.font_small. \
                render(f"{floor:01d}", True, "dark red" if floor == self.target_floor else "black")
            text_rect: pygame.Rect = text_surface.get_rect()

            text_surface = \
                pygame.transform.scale(
                    text_surface,
                    (text_surface.get_width() * sw, text_surface.get_height() * sh))

            x, y = Conf.screen_size
            width = self.position[0]
            width_offset = x / 16
            height = self.position[1]
            height_offset = y/21 * index + y/20

            text_rect.center = (width_offset + width, height_offset + height)
            game_display.blit(text_surface, text_rect)
