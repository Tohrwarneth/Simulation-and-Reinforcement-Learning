import pygame
from enums import Direction
from logic.person import Person
from utils import Conf


class GuiElevatorFloor:
    """
    GUI Representation of People on current floor of an elevator
    """
    index: int
    call: tuple[list[list[Person]], list[list[Person]]]
    currentFloor: int
    position: tuple[float, float]

    def __init__(self, index: int, current_floor: int, call_up: list[list], call_down: list[list]):
        self.index = index
        self.call = (call_up, call_down)
        self.currentFloor = current_floor

        self.position = (0, 0)

        self.update_screen_scale()

    def update(self, current_floor: int) -> None:
        """
        Update non referenced variable
        :param current_floor: current floor of the elevator
        :return: None
        """
        self.currentFloor = current_floor

    def render(self, game_display: pygame.Surface) -> None:
        """
        Rendering GUI
        :param game_display: master surface
        :return: None
        """
        self.render_floor(game_display, Direction.UP)
        self.render_floor(game_display, Direction.DOWN)

    def render_floor(self, game_display, direction: Direction) -> None:
        """
        Rendering up/down call of current floor
        :param game_display: master surface
        :param direction: up or down call
        :return: None
        """
        sw, sh = Conf.screenScale
        floor: list[Person] = self.call[direction.value][self.currentFloor]
        if len(floor) > 0:
            image_person: pygame.Surface = pygame.image.load('images/Person.png')
            image_person.convert()
            image_person = \
                pygame.transform.scale(
                    image_person,
                    (image_person.get_width() * 1.5 * sw, image_person.get_height() * 1.5 * sh))

            x, y = Conf.screenSize
            width: float = self.position[0]
            if direction == 1:
                offset_width: float = -x / 10
            else:
                offset_width: float = -x / 6
            height = self.position[1]

            person_rect: pygame.Rect = image_person.get_rect()
            person_rect.center = (offset_width + width, height)

            if direction == 0:
                image_arrow: pygame.Surface = pygame.image.load('images/Pfeil_hoch.png')
            else:
                image_arrow: pygame.Surface = pygame.image.load('images/Pfeil_runter.png')
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

            text_surface: pygame.Surface = Conf.fontSmall.render(
                str(len(floor)), True, "black")

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

    def update_screen_scale(self) -> None:
        """
        Update position and font according to the new resolution
        :return: None
        """
        screen_size: tuple[float, float] = Conf.screenSize
        offset: float = screen_size[0] / 2.858
        width: float = offset + (screen_size[0] / 3.55) * self.index
        height: float = (screen_size[1] / 2)
        self.position = (width, height)
