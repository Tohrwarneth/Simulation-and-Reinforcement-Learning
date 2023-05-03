import sys

from ui.GuiManager import GuiManager


class Simulation:
    guiActivated: bool = True
    guiManager: GuiManager

    def __int__(self):
        self.init()

    def init(self):
        if (self.guiActivated):
            self.guiManager = GuiManager()
            self.guiManager.init()

    def run(self):
        running: bool = True

        while running:
            if self.guiActivated:
                running = self.guiManager.frame()

        self.shutdown()

    def shutdown(self):
        self.guiManager.shutdown()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    simulation = Simulation()
    simulation.init()

    for param in sys.argv:
        if param.__contains__("ui="):
            value = param[3:]
            simulation.guiActivated = value != "false" and value != "False"

    simulation.run()
