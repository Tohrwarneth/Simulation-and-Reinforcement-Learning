import pygame
from src.utils import Conf

"""
Zeichnet die die Passagiere eines Aufzugs
"""
class GuiPassengers:

    def __init__(self, index: int):
        self.index: int = index
        self.position: tuple[float, float] = (0, 0)
        self.number_passengers: int = 0

        self.update_screen_scale()

    def update(self, number_passengers: int) -> None:
        self.number_passengers = number_passengers

    def render(self, game_display: pygame.Surface) -> None:
        sw, sh = Conf.screen_scale
        for i in range(0, self.number_passengers):
            image = pygame.image.load('images/Person.png')
            image.convert()

            image = \
                pygame.transform.scale(
                    image,
                    (image.get_width() * sw, image.get_height() * sh))

            offset = Conf.screenSize[0] / 2.9
            width = self.position[0]
            height = self.position[1]
            if not i == 2:
                offset += width / 10 if i % 2 == 0 else (width / -10)
                height += height / 6 if i > 2 else (height / -12)

            image_rect: pygame.Rect = image.get_rect()
            image_rect.center = (offset + width * self.index, height)

            game_display.blit(image, image_rect)

    def update_screen_scale(self) -> None:
        screen_size = Conf.screenSize
        width = (screen_size[0] / 3.55)
        height = (screen_size[1] / 2)
        self.position = (width, height)
