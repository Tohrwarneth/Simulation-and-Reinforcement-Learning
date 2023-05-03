import pygame

from IGuiObject import IGuiObject
from typing import List


class GuiManager:
    guiObjects: List[IGuiObject]

    def __int__(self):
        print('ini')
        self.guiObjects = list()

    def init(self, screen:pygame.Surface):
        for gui in self.guiObjects:
            gui.init(screen)

    def update(self, delta: int):
        for gui in self.guiObjects:
            gui.update(delta)

    def render(self):
        for gui in self.guiObjects:
            gui.render()

    def addGuiObject(self, object: IGuiObject) -> None:
        self.guiObjects.append(object)
