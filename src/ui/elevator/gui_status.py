import pygame
from utils import Conf
from enums import ElevatorState, Direction


class GUIStatus:
    """
    GUI Representation of state, position, direction and target floor of an elevator
    """
    elevatorIndex: int
    currentFloor: int
    targetFloor: int
    targetFloor: int
    state: ElevatorState
    direction: Direction
    position: tuple[float, float]

    def __init__(self, index: int, current_floor: int, state: ElevatorState, direction: Direction):
        self.elevatorIndex = index
        self.currentFloor = current_floor
        self.targetFloor = current_floor
        self.state = state
        self.direction = direction

        self.position = (0, 0)

        self.update_screen_scale()

    def update(self, current_floor: int, target_floor: int, state: ElevatorState,
               direction: Direction) -> None:
        """
        Update non referenced variable
        :param current_floor: current position of the elevator
        :param target_floor: target floor of the elevator
        :param state: state of the elevator
        :param direction: a-/descending
        :return:
        """
        self.currentFloor = current_floor
        self.targetFloor = target_floor
        self.state = state
        self.direction = direction

    def render(self, game_display: pygame.Surface) -> None:
        """
        Rendering GUI
        :param game_display: master surface
        :return: None
        """
        (sw, sh) = Conf.screenScale

        text_surface: pygame.Surface = Conf.fontSmall. \
            render(f"{self.currentFloor + 1:01d}", True, "black")

        text_rect: pygame.Rect = text_surface.get_rect()
        text_rect.center = self.position
        game_display.blit(text_surface, text_rect)

        if self.direction == Direction.UP:
            image: pygame.Surface = pygame.image.load('images/Pfeil_hoch.png')
            image.convert()
        else:
            image: pygame.Surface = pygame.image.load('images/Pfeil_runter.png')
            image.convert()

        image = \
            pygame.transform.scale(
                image,
                (image.get_width() * sw, image.get_height() * sh))

        x, _ = Conf.screenSize
        width: float = self.position[0]
        offset: float = x / 50
        height: float = self.position[1] - 5

        image_rect: pygame.Rect = image.get_rect()
        image_rect.center = (offset + width, height)
        game_display.blit(image, image_rect)

        if self.state == ElevatorState.WAIT:
            state_text: str = 'Warten'
        elif self.state == ElevatorState.UP:
            state_text: str = 'Hoch'
        elif self.state == ElevatorState.DOWN:
            state_text: str = 'Runter'
        else:
            state_text: str = self.state.name
        state_surface = Conf.fontSmall. \
            render(f"{state_text}", True, "black")

        x, _ = Conf.screenSize
        width = self.position[0]
        offset = x / 30
        height = self.position[1]

        state_rect: pygame.Rect = state_surface.get_rect()
        state_rect.center = (width - offset, height)
        game_display.blit(state_surface, state_rect)

        text_surface = Conf.fontSmall. \
            render(f"Ziel: {self.targetFloor + 1}", True, "black")

        width = self.position[0]
        offset = x / 20
        height = self.position[1]

        text_rect = text_surface.get_rect()
        text_rect.center = (width + offset, height)
        game_display.blit(text_surface, text_rect)

    def update_screen_scale(self) -> None:
        """
        Update position and font according to the new resolution
        :return: None
        """
        screen_size: list[float, float] = Conf.screenSize
        offset: float = screen_size[0] / 2.858
        width: float = offset + (screen_size[0] / 3.55) * self.elevatorIndex
        height: float = (screen_size[1] / 2.62)
        self.position = (width, height)
