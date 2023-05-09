from ..ILogicObject import ILogicObject
from ..states import DayState


class Person:
    homeFloor: int
    targetFloor: int
    currentFloor: int
    dayState: DayState

    def __init__(self, targetFloor: int):
        self.targetFloor = targetFloor
        self.homeFloor = targetFloor
        self.currentFloor = 0
        self.dayState = DayState.MORNING

    def setCurrentFloor(self, floor: int):
        self.currentFloor = floor
        if floor == self.targetFloor:
            if not self.dayState == DayState.EVENING:
                # Wenn nicht auf dem Weg nach Hause, dann gehe einen Schritt weiter (bsp. Mittagessen)
                self.dayState += 1
                if self.dayState == DayState.EVENING:
                    # Gehe nach Hause
                    self.targetFloor = 0
                elif self.dayState == DayState.PRE_LUNCH:
                    # Gehe Mittagessen
                    self.targetFloor = 10 if floor >= 10 or floor >= 7 else 5
