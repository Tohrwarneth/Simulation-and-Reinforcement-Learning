import pygame
from simpy import Environment

from src.utils import Conf, Clock


class GuiClock:

    def __init__(self):
        self.text: str = "00:00"
        self.position: tuple[float, float] = (0, 0)
        self.update_screen_scale()

    def update(self, env: Environment) -> None:
        # TODO: Hack entfernen!
        # sec = Clock.tact
        sec = env.now
        m = int(sec % 60)
        h = int(sec / 60)
        self.text = f"{h:02d}:{m:02d}"

    def render(self, game_display: pygame.Surface) -> None:
        # self.update()
        sw, sh = Conf.screenScale
        text_surface: pygame.Surface = Conf.font.render(self.text, True, "black")

        text_surface = \
            pygame.transform.scale(
                text_surface,
                (text_surface.get_width() * sw, text_surface.get_height() * sh))

        text_rect: pygame.Rect = text_surface.get_rect()
        text_rect.center = self.position
        game_display.blit(text_surface, text_rect)

        scale_surface = Conf.font.render(f"Geschwindigkeit: {Clock.speedScale}", True, "black")
        scale_surface = \
            pygame.transform.scale(
                scale_surface,
                (scale_surface.get_width() * sw, scale_surface.get_height() * sh))

        game_display.blit(scale_surface, (5, 5))

    def update_screen_scale(self):
        screen_size = Conf.screenSize
        width = screen_size[0] / 1.863
        height = screen_size[1] / 12.5
        self.position = (width, height)
