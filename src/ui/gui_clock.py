import pygame
from simpy import Environment

from src.utils import Conf, Clock


class GuiClock:

    def __init__(self):
        self.flashTick: float = 0.6
        self.remainingFlashTick: float = 0
        self.flashState: bool = True  # Visible = True, Hidden = False
        self.time_text: str = "00:00"
        self.position: tuple[float, float] = (0, 0)
        self.update_screen_scale()

    def update(self) -> None:
        sec = Clock.tact
        m = int(sec % 60)
        h = int(sec / 60)
        self.time_text = f"{h:02d}:{m:02d}"

    def render(self, game_display: pygame.Surface) -> None:
        self.update()

        clock_surface: pygame.Surface = Conf.fontLarge.render(self.time_text, True, "black")

        clock_rect: pygame.Rect = clock_surface.get_rect()
        clock_rect.center = self.position
        if self.flashState:
            game_display.blit(clock_surface, clock_rect)

        if Clock.pause:
            self.remainingFlashTick -= Clock.deltaTime
            if self.remainingFlashTick <= 0:
                self.remainingFlashTick = self.flashTick
                self.flashState = not self.flashState

            pause_surface: pygame.Surface = Conf.fontLarge.render("Pause", True, "black")

            pause_rect: pygame.Rect = pause_surface.get_rect()
            width, height = self.position
            height_offset = Conf.screenSize[1] / 15
            pause_rect.center = (width, height + height_offset)
            if not self.flashState:
                game_display.blit(pause_surface, pause_rect)
        else:
            self.remainingFlashTick = 0.0
            self.flashState = True

        status_text: str = f"Geschwindigkeit (+/-): {Clock.speedScale if not Clock.pause else Clock.speedPrePaused}"
        scale_surface = Conf.fontLarge.render(status_text, True, "black")

        game_display.blit(scale_surface, (5, 5))

    def update_screen_scale(self):
        screen_size = Conf.screenSize
        width = screen_size[0] / 1.863
        height = screen_size[1] / 12.5
        self.position = (width, height)
