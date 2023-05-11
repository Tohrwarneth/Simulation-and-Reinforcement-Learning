import pygame
from src.conf import Conf


class GuiFloor:
    position: tuple[float, float]
    person_floor: list[tuple[int, int]]

    def __init__(self):
        self.update_screen_scale()

    def init(self):
        pass

    def update(self, person_floor: list[tuple[int, int]]) -> None:
        # self.person_floor = person_floor
        pass

    def render(self, game_display: pygame.Surface) -> None:
        self.drawFloor(game_display, 0)
        self.drawFloor(game_display, 1)

    def drawFloor(self, game_display, direction: int):
        sw, sh = Conf.screen_scale
        for index, floor in enumerate(self.person_floor):
            if floor[direction] > 0:
                if direction == 0:
                    image_arrow = pygame.image.load('images/Pfeil_hoch.png')
                else:
                    image_arrow = pygame.image.load('images/Pfeil_runter.png')
                image_arrow.convert()

                image_arrow = \
                    pygame.transform.scale(
                        image_arrow,
                        (image_arrow.get_width() * 1.5 * sw, image_arrow.get_height() * 1.5 * sh))

                width, height = self.position
                if direction == 1:
                    offset_width = width/10
                else:
                    offset_width = 0
                offset_height = height/3.15 * index

                arrow_rect: pygame.Rect = image_arrow.get_rect()
                arrow_rect.center = (offset_width + width, offset_height + height)

                game_display.blit(image_arrow, arrow_rect)

    def update_screen_scale(self):
        screen_size = Conf.screen_size
        width = screen_size[0] / 12
        height = screen_size[1] / 6.5
        self.position = (width, height)

    def add_person_number(self):
        self.person_number += 1

    def remove_person_number(self):
        self.person_number -= 1
