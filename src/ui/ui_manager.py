import math
import time

import pygame
import simpy

from src import Conf
from src.logic.elevator import Elevator


class UIManager:
    env: simpy.Environment
    elevatorList: list[Elevator]
    numOfElevators: int
    queUpward: list[list]
    queDownward: list[list]

    def __init__(self, env: simpy.Environment, elevator_list: list[Elevator],
                 que_upward: list[list], que_downward: list[list]):
        self.env = env
        self.elevatorList = elevator_list
        self.numOfElevators = len(elevator_list)
        self.queUpward = que_upward
        self.queDownward = que_downward

    def draw(self):
        pygame.init()
        screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        # get window size
        width, height = pygame.display.get_surface().get_size()

        pygame.display.set_caption('Elevator Simulation')

        running = True
        timePerStep = 0.01

        # Colors
        black = (0, 0, 0)
        green = (34, 139, 34)
        red = (139, 0, 0)

        # Fonts
        bigFont = pygame.font.Font(pygame.font.get_default_font(), 20)  # fontsize = 20
        smallFont = pygame.font.Font(pygame.font.get_default_font(), 14)  # fontsize = 20
        pause = False
        while running:
            # poll for events
            # pygame.QUIT event means the user clicked X to close your window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.WINDOWRESIZED:
                    width, height = screen.get_width(), screen.get_height()
                if event.type == pygame.KEYDOWN:

                    # space to pause
                    if (event.key == pygame.K_SPACE):
                        if (pause):
                            pause = False
                        else:
                            pause = True

                    # press UP to double the speed
                    if (event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS):
                        timePerStep /= 2
                    # press Down to half the speed
                    if (event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS):
                        timePerStep *= 2

            if pause:
                continue

            # fill the screen with a color to wipe away anything from last frame
            screen.fill("white")

            # draw glock
            hours, min = divmod(self.env.now, 60)
            sec = int((min % 1) * 1000)
            text = bigFont.render('{:02d}:{:02d}.{:03d}'.format(*divmod(math.floor(self.env.now), 60), sec), True,
                                  black)  # FG, BG
            screen.blit(text, (width / 2, 10))

            # draw legend

            text = smallFont.render('Space = Stop', True, black)
            screen.blit(text, (width - 200, 10))

            text = smallFont.render('Up/Down = Speed + / -', True, black)
            screen.blit(text, (width - 200, 30))
            # Columns
            pygame.draw.rect(screen, black, (0, 0, width / 8, height), 3)  # black = (R,G,B) = (0,0,0)
            pygame.draw.rect(screen, black, (0, 0, width / 8 * 1 / 3, height), 3)  # black = (R,G,B) = (0,0,0)
            pygame.draw.rect(screen, black, (0, 0, width / 8 * 2 / 3, height), 3)  # black = (R,G,B) = (0,0,0)
            # Header
            headerHeight = 30
            pygame.draw.rect(screen, black, (0, 0, width / 8, headerHeight), 3)  # black = (R,G,B) = (0,0,0)

            textPaddingX = 5
            textPaddingY = 10

            # Text Col Header Floor
            text = smallFont.render('Floor', True, black)
            screen.blit(text, (0 + textPaddingX, textPaddingY))

            # Text Col Header Up
            text = smallFont.render('UP', True, green)
            screen.blit(text, (width / 8 * 1 / 3 + textPaddingX, textPaddingY))

            # Text Col Header Down
            text = smallFont.render('Down', True, red)
            screen.blit(text, (width / 8 * 2 / 3 + textPaddingX, textPaddingY))

            floorBoxHeights = (height - headerHeight) / Conf.max_floor

            # rows for each floor
            for i in range(Conf.max_floor):
                # seperaters
                pygame.draw.line(screen, black, (0, headerHeight + floorBoxHeights * (i + 1)),
                                 (width / 8, headerHeight + floorBoxHeights * (i + 1)))
                # Floornumber
                text = smallFont.render(f'{Conf.max_floor - 1 - i}', True, black)
                screen.blit(text, (0 + textPaddingX, headerHeight + floorBoxHeights * i + textPaddingY))
                # Que UP
                text = smallFont.render(f'{len(self.queUpward[Conf.max_floor - 1 - i])}', True, green)
                screen.blit(text, (width / 8 * 1 / 3 + textPaddingX, headerHeight + floorBoxHeights * i + textPaddingY))
                # Que Dowm
                text = smallFont.render(f'{len(self.queDownward[Conf.max_floor - 1 - i])}', True, red)
                screen.blit(text, (width / 8 * 2 / 3 + textPaddingX, headerHeight + floorBoxHeights * i + textPaddingY))

            elevatorPaddingX = 20
            elevatorHeight = width / 4
            elevatorWidth = width / 4
            startPosX = width / 8 + elevatorPaddingX * 2

            for i in range(self.numOfElevators):
                # draw elevator
                pygame.draw.rect(screen, black, (
                    startPosX + i * (elevatorWidth + elevatorPaddingX), height / 2 - elevatorHeight / 2, elevatorWidth,
                    height / 2), 3)  # black = (R,G,B) = (0,0,0)
                # current floor
                text = smallFont.render(f'Pos: {self.elevatorList[i].position}', True, black)
                screen.blit(text, (
                    startPosX + i * (elevatorWidth + elevatorPaddingX) + elevatorWidth / 2 - textPaddingX,
                    height / 2 - elevatorHeight / 2 + 2 * textPaddingY))

                # Draw Direction
                if (self.elevatorList[i].direction == 1):

                    text = smallFont.render(f'UP', True, green)
                    screen.blit(text, (
                        startPosX + i * (elevatorWidth + elevatorPaddingX) + elevatorWidth / 2 - textPaddingX,
                        height / 2 - elevatorHeight / 2 - 2 * textPaddingY))
                elif (self.elevatorList[i].direction == -1):
                    text = smallFont.render(f'DOWN', True, red)
                    screen.blit(text, (
                        startPosX + i * (elevatorWidth + elevatorPaddingX) + elevatorWidth / 2 - textPaddingX,
                        height / 2 - elevatorHeight / 2 - 2 * textPaddingY))
                else:
                    text = smallFont.render(f'WAITING', True, black)
                    screen.blit(text, (
                        startPosX + i * (elevatorWidth + elevatorPaddingX) + elevatorWidth / 2 - textPaddingX,
                        height / 2 - elevatorHeight / 2 - 2 * textPaddingY))

                # List Passengers and their choosen floor
                for j, p in enumerate(self.elevatorList[i].passengers):
                    text = smallFont.render(f'Person {p.id} -> {p.schedule[0][1]}', True, black)
                    screen.blit(text, (
                        startPosX + i * (elevatorWidth + elevatorPaddingX) + elevatorWidth / 2 - textPaddingX * 5,
                        height / 2 + 2 * textPaddingY * j))

            # flip() the display to put your work on screen
            pygame.display.flip()
            time.sleep(timePerStep)
            # clock.tick(60)  # limits FPS to 60
            yield self.env.timeout(Conf.deltaTime)
