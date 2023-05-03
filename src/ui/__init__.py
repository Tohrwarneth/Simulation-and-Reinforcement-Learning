# Example file showing a circle moving on screen
import pygame
from GuiManager import GuiManager

guiManager: GuiManager = GuiManager()

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

# fill the screen with a color to wipe away anything from last frame
screen.fill("#F6E8C3")

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

guiManager.init(screen)

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    guiManager.update(dt)


    guiManager.render()
    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()
