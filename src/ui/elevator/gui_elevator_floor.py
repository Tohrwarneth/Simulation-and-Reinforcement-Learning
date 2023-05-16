import pygame

from src.enums import Direction
from src.logic.person import Person
from src_old.conf import Conf


class GuiElevatorFloor:
    index: int

    def __init__(self, index: int, call_up: list[list], call_down: list[list]):
        self.index = index
        self.call: tuple[list[list[Person]], list[list[Person]]] = (call_up, call_down)

        self.position: tuple[float, float] = (0, 0)

        self.update_screen_scale()

    def render(self, game_display: pygame.Surface) -> None:
        self.drawFloor(game_display, Direction.UP)
        self.drawFloor(game_display, Direction.DOWN)

    def drawFloor(self, game_display, direction: Direction):
        sw, sh = Conf.screen_scale
        if self.call[direction.value] > 0:
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
                str(len(self.call[direction.value])), True, "black")
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
