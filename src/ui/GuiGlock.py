import pygame

from src.conf import Conf
from src.glock import Glock
from src.ui.IGuiObject import IGuiObject


class GuiGlock(IGuiObject):
    position: tuple[float, float]
    text: str = "00:00"

    def __init__(self):
        self.update_screen_scale()

    def init(self) -> None:
        pass

    def update(self, delta_time: float) -> None:
        h, m = Glock.get_time()
        self.text = f"{h:02d}:{m:02d}"

    def render(self, game_display: pygame.Surface) -> None:
        text_surface: pygame.Surface = Conf.font.render(self.text, True, "black")
        text_rect: pygame.Rect = text_surface.get_rect()
        text_rect.center = self.position
        game_display.blit(text_surface, text_rect)

        scale_surface = Conf.font.render(f"Speed: {Conf.speed_scale}", True, "black")
        game_display.blit(scale_surface, (5, 5))

    def update_screen_scale(self):
        screen_size = Conf.screen_size
        width = screen_size[0] / 1.863
        height = screen_size[1] / 12.5
        self.position = (width, height)