import simpy
from matplotlib import pyplot as plt
from simpy.resources.store import StoreGet

from src_old.logic.elevator.Elevator import Elevator
from src_old.logic.person.Person import Person
from src_old.logic.person.person_manager import PersonManager
from src_old.ui.GuiFloor import GuiFloor
from src_old.conf import LogData, Conf
from .states import ElevatorState, Direction
from src_old.clock import Clock


class LogicManager:
    env: simpy.Environment
    person_manager: PersonManager
    elevators: list[Elevator]

    def __init__(self, env: simpy.Environment, gui: GuiFloor = None):
        self.env = env
        self.person_manager = PersonManager(env)
        self.person_manager.init(gui=gui)
        self.elevators = list()

    def update(self) -> LogData:
        log_data: LogData = LogData(int(self.env.now))
        data: list = list()
        # data += self.person_manager.update(tact)

        data += self.person_manager.update()
        for elevator in self.elevators:
            data += elevator.update(self.person_manager.get_person_on_floor(elevator.currentFloor))
            self.check_for_passengers(elevator)
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
            person.next_job()

    def load_elevator(self, elevator: Elevator, direction: Direction, amount: int):
        passengers: list[Person] = list()
        for i in range(amount):
            if direction == Direction.UP:
                persons = self.person_manager.person_floor_up[elevator.currentFloor].items
                if persons:
                    passengers.append(persons.pop(0))
        return passengers

    def check_for_passengers(self, elevator: Elevator):
        if elevator.direction_state == ElevatorState.DOWN:
            next_floor = self.person_manager.next_down(elevator.get_next_floor())
            if not next_floor == None:
                if elevator.targetFloor < next_floor:
                    elevator.targetFloor = next_floor
                else:
                    elevator.targetFloor = next_floor
        else:
            next_floor = self.person_manager.next_above(elevator.get_next_floor())
            if not next_floor == None:
                if elevator.targetFloor > next_floor:
                    elevator.targetFloor = next_floor
            else:
                if elevator.direction_state == ElevatorState.WAIT:
                    elevator.direction_state = ElevatorState.DOWN

    def eod(self):
        print("Person remaining in Building", Conf.total_amount_person - self.person_manager.leaved_person)
        # Draw Spawn Gamma
        # if Conf.show_plots:
        #     plt.plot([j for j in range(0, 24 * 60)], gamma, linewidth=2, color='r')
        #     plt.xlabel('Zeit ins Sekunden')
        #     plt.ylabel('Gamma')
        #     plt.show()
        #     plt.savefig(f'{Conf.plot_path}/Gebaeude_Eingang.png')
