from .elevator.Elevator import Elevator
from .person.person_manager import PersonManager
from src.ui.GuiFloor import GuiFloor
from src.conf import LogData


class LogicManager:
    person_manager: PersonManager
    elevators: list[Elevator]

    def __init__(self, gui_floor: GuiFloor = None):
        self.person_manager = PersonManager()
        self.person_manager.init(gui=gui_floor)
        self.elevators = list()
        self.person_floor = self.person_manager.person_floor

    def update(self, tact: int) -> LogData:
        log_data: LogData = LogData(tact)
        data: list = list()
        data += self.person_manager.update(tact)

        for elevator in self.elevators:
            data += elevator.update(tact, self.person_manager.person_floor[elevator.currentFloor])

        log_data.add_data(data)
        return log_data

    def add_elevator(self, elevator):
        self.elevators.append(elevator)

    def get_log_header(self) -> list[str]:
        log_header: list[str] = list()
        log_header += self.person_manager.get_log_header()
        for elevator in self.elevators:
            log_header += elevator.get_log_header()

        return log_header
