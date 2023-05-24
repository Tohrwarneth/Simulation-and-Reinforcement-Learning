import pygame
from utils import Conf, Clock
from logic.elevator import Elevator
from ui.gui_clock import GuiClock
from ui.gui_floor import GuiFloor
from ui.elevator.gui_elevator import GuiElevator


class GuiManager:
    """
    GUI Renderer and Keybinding
    """
    elevatorList: list[Elevator]
    numOfElevators: int
    callUp: list[list]
    callDown: list[list]
    speedPrePaused: float

    clock: GuiClock
    floor: GuiFloor
    elevatorGuiList: list[GuiElevator]
    background_image: pygame.Surface

    def __init__(self, elevator_list: list[Elevator],
                 call_up: list[list], call_down: list[list]):
        self.elevatorList = elevator_list
        self.numOfElevators = len(elevator_list)
        self.callUp = call_up
        self.callDown = call_down

        self.clock = GuiClock()
        self.floor = GuiFloor(call_up=self.callUp, call_down=self.callDown)
        self.elevatorGuiList = list()
        for i in range(self.numOfElevators):
            self.elevatorGuiList.append(GuiElevator(self.elevatorList[i]))

        pygame.init()
        self.screen = pygame.display.set_mode(Conf.screenSize, pygame.RESIZABLE)

        Conf.screenScale = (Conf.screenSize[0] / Conf.screenOriginSize[0]
                            , Conf.screenSize[1] / Conf.screenOriginSize[1])

        self.screen.fill("pink")
        pygame.display.set_caption('Elevator Simulator')

        self.update_screen_scale()

    def render(self) -> None:
        """
        Rendering GUI
        :return: None
        """
        for event in pygame.event.get():
            # Beenden
            if event.type == pygame.QUIT:
                Clock.running = False
            # Beenden
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    Clock.running = False
                # Pause
                if event.key == pygame.K_SPACE:
                    Clock.pause = not Clock.pause
                    if Clock.pause:
                        Clock.speedPrePaused = Clock.speedScale
                        Clock.speedScale = 1.0
                    else:
                        Clock.speedScale = Clock.speedPrePaused
                # Beschleunigen
                if event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                    if Clock.pause:
                        Clock.speedPrePaused *= 2
                        if Clock.skip:
                            Clock.skip = None
                            Clock.speedPrePaused = 1.0
                    else:
                        if Clock.skip:
                            Clock.skip = None
                            Clock.speedScale = 1.0
                        Clock.speedScale *= 2
                # Verlangsamen
                if event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    if Clock.pause:
                        Clock.speedPrePaused /= 2
                        if Clock.skip:
                            Clock.skip = None
                            Clock.speedPrePaused = 1.0
                    else:
                        if Clock.skip:
                            Clock.skip = None
                            Clock.speedScale = 1.0
                        Clock.speedScale /= 2
            if event.type == pygame.VIDEORESIZE:
                width: float = event.w
                height: float = event.h
                Conf.screenSize = (width, height)
                Conf.screenScale = (width / Conf.screenOriginSize[0]
                                    , height / Conf.screenOriginSize[1])
                self.update_screen_scale()
                self.screen = pygame.display.set_mode((width, height),
                                                      pygame.RESIZABLE)
                self.update_screen_scale()

        self.screen.blit(self.background_image, (0, 0))

        self.clock.render(self.screen)
        self.floor.render(self.screen)
        for elevator in self.elevatorGuiList:
            elevator.render(self.screen)

        pygame.display.flip()

    def update_screen_scale(self) -> None:
        """
        Update position and font according to the new resolution
        :return: None
        """
        sw, sh = Conf.screenScale
        Conf.fontLarge = pygame.font.Font(pygame.font.get_default_font(), int(Conf.fontSizeLarge * sh))  # 30
        Conf.fontSmall = pygame.font.Font(pygame.font.get_default_font(), int(Conf.fontSizeSmall * sh))  # 20

        self.background_image = pygame.image.load('images/FahrstuhlLayout.png')
        self.background_image.convert()
        self.background_image = pygame.transform.scale(
            self.background_image, Conf.screenSize)

        self.clock.update_screen_scale()
        self.floor.update_screen_scale()
        for elevator in self.elevatorGuiList:
            elevator.update_screen_scale()
