import simpy
from matplotlib import pyplot as plt

from src.logic.elevator.Elevator import Elevator
from src.logic.person.Person import Person
from src.logic.person.person_manager import PersonManager
from src.ui.GuiFloor import GuiFloor
from src.conf import LogData, Conf
from .states import ElevatorState


class LogicManager:
    env: simpy.Environment
    person_manager: PersonManager
    elevators: list[Elevator]

    def __init__(self, env: simpy.Environment, gui: GuiFloor = None):
        self.env = env
        self.person_manager = PersonManager(env)
        self.person_manager.init(gui=gui)
        self.elevators = list()

    def update(self, tact: int) -> LogData:
        log_data: LogData = LogData(tact)
        data: list = list()
        data += self.person_manager.update(tact)

        for elevator in self.elevators:
            self.check_for_passengers(elevator)
            data += elevator.update(tact,
                                    self.person_manager.get_person_on_floor(elevator.currentFloor))

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

    def reached_floor(self, passangers: list[Person]):
        for person in passangers:
            self.person_manager.reached_target(person)

    def load_elevator(self, event):
        pass

    def check_for_passengers(self, elevator: Elevator):
        if elevator.state == ElevatorState.WAIT:
            elevator.state = ElevatorState.UP
            self.check_for_passengers(elevator)
        elif elevator.state == ElevatorState.UP:
            next_floor = self.person_manager.next_above(elevator.get_next_floor())
            if next_floor == None:
                if len(elevator.jobs) == 0:
                    elevator.state = ElevatorState.DOWN
                    self.check_for_passengers(elevator)
        elif elevator.state == ElevatorState.DOWN:
            next_floor = self.person_manager.next_down(elevator.get_next_floor())
            if next_floor == None:
                if len(elevator.jobs) == 0:
                    elevator.state = ElevatorState.WAIT
        else:
            return

    def eod(self):
        gamma = self.person_manager.gamma
        print(sum(gamma[0:650]))
        print(sum(gamma[650:800]))
        # Draw Spawn Gamma
        if Conf.show_plots:
            plt.plot([j for j in range(0, 24 * 60)], gamma, linewidth=2, color='r')
            plt.xlabel('Zeit ins Sekunden')
            plt.ylabel('Gamma')
            plt.show()
            plt.savefig(f'{Conf.plot_path}/Gebaeude_Eingang.png')
