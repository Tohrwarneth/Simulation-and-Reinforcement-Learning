import pygame
from src.conf import Conf


class GuiFloor:
    index: int
    position: tuple[float, float]
    person_number: tuple[int, int] = (0, 0)

    def __init__(self, index: int):
        self.index = index
        self.update_screen_scale()

    def init(self):
        pass

    def update(self, person_floor: tuple[int, int]) -> None:
        self.person_number = person_floor

    def render(self, game_display: pygame.Surface) -> None:
        self.drawFloor(game_display, 0)
        self.drawFloor(game_display, 1)

    def drawFloor(self, game_display, direction: int):
        sw, sh = Conf.screen_scale
        if self.person_number[direction] > 0:
            image_person = pygame.image.load('images/Person.png')
            image_person.convert()
            image_person = \
                pygame.transform.scale(
                    image_person,
                    (image_person.get_width() * 1.5 * sw, image_person.get_height() * 1.5 * sh))

            x, y = Conf.screen_size
            width = self.position[0]
            if direction == 1:
                offset_width = -x / 10
            else:
                offset_width = -x / 6
            height = self.position[1]

            person_rect: pygame.Rect = image_person.get_rect()
            person_rect.center = (offset_width + width, height)

            if direction == 0:
                image_arrow = pygame.image.load('images/Pfeil_hoch.png')
            else:
                image_arrow = pygame.image.load('images/Pfeil_runter.png')
            image_arrow.convert()
            image_arrow = \
                pygame.transform.scale(
                    image_arrow,
                    (image_arrow.get_width() * sw, image_arrow.get_height() * sh))

            if direction == 1:
                offset_width = -x / 11.5
            else:
                offset_width = -x / 6.5
            offset_height = -x / 45

            arrow_rect: pygame.Rect = image_arrow.get_rect()
            arrow_rect.center = (offset_width + width, offset_height + height)

            text_surface: pygame.Surface = Conf.font_small.render(
                str(self.person_number[direction]), True, "black")
            text_surface = \
                pygame.transform.scale(
                    text_surface,
                    (text_surface.get_width() * sw, text_surface.get_height() * sh))

            if direction == 1:
                offset_width = -x / 11.5
            else:
                offset_width = -x / 6.5
            offset_height = -x / 120

            text_rect: pygame.Rect = text_surface.get_rect()
            text_rect.center = (offset_width + width, offset_height + height)

            game_display.blit(text_surface, text_rect)
            game_display.blit(image_arrow, arrow_rect)
            game_display.blit(image_person, person_rect)

    def update_screen_scale(self):
        screen_size = Conf.screen_size
        offset = screen_size[0] / 2.858
        width = offset + (screen_size[0] / 3.55) * self.index
        height = (screen_size[1] / 2)
        self.position = (width, height)

    def add_person_number(self):
        self.person_number += 1

    def remove_person_number(self):
        self.person_number -= 1
