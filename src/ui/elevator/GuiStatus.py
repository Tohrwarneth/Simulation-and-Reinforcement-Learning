import pygame

from src.conf import Conf


class GUIStatus:
    index: int
    currentFloor: int
    target_floor: int
    position: tuple[float, float]
    offset: float

    def __init__(self, index: int, floor: int):
        self.index = index
        self.target_floor = floor
        self.currentFloor = floor

        self.update_screen_scale()

    def update_screen_scale(self):
        screen_size = Conf.screen_size
        self.offset = screen_size[0] / 2.858
        width = self.offset + (screen_size[0] / 3.55) * self.index
        height = (screen_size[1] / 2.62)
        self.position = (width, height)

    def init(self) -> None:
        pass

    def update(self, floor: int, target_floor: int) -> None:
        self.currentFloor = floor
        self.target_floor = target_floor

    def render(self, game_display: pygame.Surface) -> None:
        text_surface: pygame.Surface = Conf.font_small. \
            render(f"{self.currentFloor:01d}", True, "black")
        text_rect: pygame.Rect = text_surface.get_rect()
        text_rect.center = self.position
        game_display.blit(text_surface, text_rect)

        floor_offset = self.target_floor - self.currentFloor
        if not floor_offset == 0:
            if floor_offset > 0:
                image = pygame.image.load('images/Pfeil_hoch.png')
                image.convert()
            else:
                image = pygame.image.load('images/Pfeil_runter.png')
                image.convert()

            x, _ = Conf.screen_size
            width = self.position[0]
            offset = x/55
            height = self.position[1] - 5

            text_rect: pygame.Rect = image.get_rect()
            text_rect.center = (offset + width, height)

            game_display.blit(image, text_rect)
