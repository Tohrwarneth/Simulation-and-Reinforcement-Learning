import pygame

from src.enums import Direction
from src.logic.person import Person
from src.utils import Conf


class GuiFloor:

    def __init__(self, call_up: list[list[Person]], call_down: list[list[Person]]):
        self.call: tuple[list[list[Person]], list[list[Person]]] = (call_up, call_down)
        self.position: tuple[float, float] = (0, 0)

        self.update_screen_scale()

    def render(self, game_display: pygame.Surface) -> None:
        self.draw_floor(game_display, Direction.UP)
        self.draw_floor(game_display, Direction.DOWN)

    def draw_floor(self, game_display, direction: Direction):
        (sw, sh) = Conf.screenScale
        for index, floor in enumerate(reversed(self.call[direction.value])):
            if len(floor) > 0:
                if direction.value == 0:
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
                    offset_width = width / 10
                else:
                    offset_width = 0
                offset_height = height / 3.15 * index

                arrow_rect: pygame.Rect = image_arrow.get_rect()
                arrow_rect.center = (offset_width + width, offset_height + height)
                game_display.blit(image_arrow, arrow_rect)

            if not (index == 14 and direction == direction.DOWN):
                text_surface: pygame.Surface = Conf.fontSmall. \
                    render(f"{len(floor):01d}", True, "black")
                text_rect: pygame.Rect = text_surface.get_rect()

                width, height = self.position
                offset_width = Conf.screenSize[0] / 30
                if direction == 1:
                    offset_width += width / 5
                else:
                    offset_width += width / 30
                offset_height = height / 3.15 * index

                text_rect.center = (offset_width + width, offset_height + height)
                game_display.blit(text_surface, text_rect)

    def update_screen_scale(self):
        screen_size = Conf.screenSize
        width = screen_size[0] / 12
        height = screen_size[1] / 6.5
        self.position = (width, height)
