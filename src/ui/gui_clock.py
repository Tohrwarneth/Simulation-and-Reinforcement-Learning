import pygame
from utils import Conf, Clock


class GuiClock:
    """
    GUI Representation of the Clock and simulation speed
    """
    flashTick: float
    remainingFlashTick: float
    isVisible: bool  # Visible = True, Hidden = False
    timeText: str
    position: tuple[float, float]

    def __init__(self):
        self.flashTick = 0.6
        self.remainingFlashTick = 0
        self.isVisible = True
        self.timeText = "00:00"
        self.position = (0, 0)

        self.update_screen_scale()

    def update(self) -> None:
        """
        Update non referenced variable
        :return: None
        """
        sec = Clock.tact
        m = int(sec % 60)
        h = int(sec / 60)
        self.timeText = f"{h:02d}:{m:02d}"

    def render(self, game_display: pygame.Surface) -> None:
        """
        Rendering GUI
        :param game_display: master surface
        :return: None
        """
        self.update()

        clock_surface: pygame.Surface = Conf.fontLarge.render(self.timeText, True, "black")

        clock_rect: pygame.Rect = clock_surface.get_rect()
        clock_rect.center = self.position
        if self.isVisible:
            game_display.blit(clock_surface, clock_rect)

        if Clock.pause:
            self.remainingFlashTick -= Clock.deltaTime
            if self.remainingFlashTick <= 0:
                self.remainingFlashTick = self.flashTick
                self.isVisible = not self.isVisible

            pause_surface: pygame.Surface = Conf.fontLarge.render("Pause", True, "#333333")

            pause_rect: pygame.Rect = pause_surface.get_rect()
            width, height = self.position
            height_offset: float = Conf.screenSize[1] / 15
            pause_rect.center = (width, height + height_offset)
            game_display.blit(pause_surface, pause_rect)
        else:
            self.remainingFlashTick = 0.0
            self.isVisible = True

        status_text: str = f"Geschwindigkeit (+/-): {Clock.speedScale if not Clock.pause else Clock.speedPrePaused}"
        scale_surface = Conf.fontLarge.render(status_text, True, "black")

        game_display.blit(scale_surface, (5, 5))

    def update_screen_scale(self):
        """
        Update position and font according to the new resolution
        :return: None
        """
        screen_size: tuple[float, float] = Conf.screenSize
        width: float = screen_size[0] / 1.863
        height: float = screen_size[1] / 12.5
        self.position = (width, height)
