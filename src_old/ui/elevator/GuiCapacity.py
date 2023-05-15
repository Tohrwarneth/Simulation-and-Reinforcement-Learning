import pygame

from src_old.conf import Conf


class GuiCapacity:
    index: int
    position: tuple[float, float]
    person_number: int = 0

    def __init__(self, index: int):
        self.index = index
        self.update_screen_scale()

    def init(self):
        pass

    def update(self, delta: float) -> None:
        pass

    def render(self, game_display: pygame.Surface) -> None:
        sw, sh = Conf.screen_scale
        for i in range(0, self.person_number):
            image = pygame.image.load('images/Person.png')
            image.convert()

            image = \
                pygame.transform.scale(
                    image,
                    (image.get_width() * sw, image.get_height() * sh))

            offset = Conf.screen_size[0] / 2.9
            width = self.position[0]
            height = self.position[1]
            if not i == 2:
                offset += width / 10 if i % 2 == 0 else (width / -10)
                height += height / 6 if i > 2 else (height / -12)

            image_rect: pygame.Rect = image.get_rect()
            image_rect.center = (offset + width * self.index, height)

            game_display.blit(image, image_rect)

    def update_screen_scale(self):
        screen_size = Conf.screen_size
        width = (screen_size[0] / 3.55)
        height = (screen_size[1] / 2)
        self.position = (width, height)

    def add_person_number(self):
        self.person_number += 1

    def remove_person_number(self):
        self.person_number -= 1
