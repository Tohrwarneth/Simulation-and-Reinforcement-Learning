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

        self.update_screen_scale()

    def update(self, current_floor: int, target_floor: int, state: ElevatorState) -> None:
        self.currentFloor = current_floor
        self.targetFloor = target_floor
        self.state = state

    def render(self, game_display: pygame.Surface) -> None:
        (sw, sh) = Conf.screenScale

        text_surface: pygame.Surface = Conf.fontSmall. \
            render(f"{self.currentFloor + 1:01d}", True, "black")

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
            offset = x / 50
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
        state_surface = Conf.fontSmall. \
            render(f"{state_text}", True, "black")

        x, _ = Conf.screenSize
        width = self.position[0]
        offset = x / 30
        height = self.position[1]

        state_rect = state_surface.get_rect()
        state_rect.center = (width - offset, height)
        game_display.blit(state_surface, state_rect)

        text_surface = Conf.fontSmall. \
            render(f"Ziel: {self.targetFloor}", True, "black")

        width = self.position[0]
        offset = x / 20
        height = self.position[1]

        text_rect = text_surface.get_rect()
        text_rect.center = (width + offset, height)
        game_display.blit(text_surface, text_rect)

    def update_screen_scale(self):
        screen_size = Conf.screenSize
        offset = screen_size[0] / 2.858
        width = offset + (screen_size[0] / 3.55) * self.index
        height = (screen_size[1] / 2.62)
        self.position = (width, height)
