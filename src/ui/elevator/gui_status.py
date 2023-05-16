import pygame
from src.utils import Conf
from src.enums import ElevatorState


class GUIStatus:

    def __init__(self, index: int, current_floor: int, state: ElevatorState):
        self.index = index
        self.currentFloor = current_floor
        self.targetFloor = current_floor
        self.state = state

        self.position: tuple[float, float] = (0, 0)
        self.offset: float = 0  # TODO: Offset wird nicht genutzt. Warum?

        self.update_screen_scale()

    def update(self, current_floor: int, target_floor: int, state: ElevatorState) -> None:
        self.currentFloor = current_floor
        self.targetFloor = target_floor
        self.state = state

    def render(self, game_display: pygame.Surface) -> None:
        (sw, sh) = Conf.screen_scale

        text_surface: pygame.Surface = Conf.font_small. \
            render(f"{self.currentFloor:01d}", True, "black")
        text_surface = \
            pygame.transform.scale(
                text_surface,
                (text_surface.get_width() * sw, text_surface.get_height() * sh))

        text_rect: pygame.Rect = text_surface.get_rect()
        text_rect.center = self.position
        game_display.blit(text_surface, text_rect)

        # TODO: Statt Richtung, Target angeben
        floor_offset = self.targetFloor - self.currentFloor
        if floor_offset != 0:
            if floor_offset > 0:
                image = pygame.image.load('images/Pfeil_hoch.png')
                image.convert()
            else:
                image = pygame.image.load('images/Pfeil_runter.png')
                image.convert()

            image = \
                pygame.transform.scale(
                    image,
                    (image.get_width() * sw, image.get_height() * sh))

            x, _ = Conf.screenSize
            width = self.position[0]
            offset = x / 55
            height = self.position[1] - 5

            image_rect: pygame.Rect = image.get_rect()
            image_rect.center = (offset + width, height)
            game_display.blit(image, image_rect)

        if self.state == ElevatorState.WAIT:
            state_text = 'Warten'
        elif self.state == ElevatorState.UP:
            state_text = 'Hoch'
        elif self.state == ElevatorState.DOWN:
            state_text = 'Runter'
        else:
            state_text = self.state
        text_surface = Conf.font_small. \
            render(f"{state_text}", True, "black")
        text_surface = \
            pygame.transform.scale(
                text_surface,
                (text_surface.get_width() * sw, text_surface.get_height() * sh))

        x, _ = Conf.screenSize
        width = self.position[0]
        offset = x / 30
        height = self.position[1]

        text_rect = text_surface.get_rect()
        text_rect.center = (width - offset, height)
        game_display.blit(text_surface, text_rect)

    def update_screen_scale(self):
        screen_size = Conf.screenSize
        self.offset = screen_size[0] / 2.858
        width = self.offset + (screen_size[0] / 3.55) * self.index
        height = (screen_size[1] / 2.62)
        self.position = (width, height)
