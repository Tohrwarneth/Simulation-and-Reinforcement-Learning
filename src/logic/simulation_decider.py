from enums import Direction, ElevatorState
from logic.decider_interface import IDecider
from logic.person import Person
from utils import Conf


class SimulationDecider(IDecider):
    @staticmethod
    def search_for_call(position: int, direction: Direction, call_up: list[list[Person]],
                        call_down: list[list[Person]]) -> tuple[int, Direction] | tuple[None, Direction]:
        """
        Searchs the next requested floor from the current position following the elevator's direction.
        If building end reached turn direction and search again.
        :return: Requested floor or None if no requests exists
        """

        call: list[Person]

        searched_one_direction: bool = False

        for _ in range(2):
            # 4 cases per direction:
            #
            # if on the way up: check if anyone from position to 14 wants to go up.
            # if on the way up: check if anyone from 14 to position wants to go down.
            # switch direction: up -> down
            # if on the way up: check if anyone from position to 0 wants to go down.
            # if on the way up: check if anyone from 0 to position wants to go up.
            #
            # if on the way down: check if anyone from position to 0 wants to go down.
            # if on the way down: check if anyone from 0 to position wants to go up.
            # switch direction: down -> up
            # if on the way down: check if anyone from position to 14 wants to go up.
            # if on the way down: check if anyone from 14 to position wants to go down.

            # if already on the buildings end, only one direction have to be checked
            if position == Conf.maxFloor - 1:
                searched_one_direction = True
            elif position == 0:
                searched_one_direction = True

            search_range = (range(position, Conf.maxFloor), range(0, position + 1))
            if direction == Direction.UP:
                for i in search_range[0]:  # position -> 14
                    if call_up[i]:
                        return i, direction
                for i in reversed(search_range[0]):  # 14 -> position
                    if call_down[i]:
                        # if current floor requested follow the request to go down
                        if position == i:
                            direction = Direction.DOWN
                        return i, direction
                # change direction if requested: up -> down
                for i in reversed(search_range[1]):  # position -> 0
                    if call_down[i]:
                        direction = Direction.DOWN
                        return i, direction
                for i in search_range[1]:  # 0 -> position
                    if call_up[i]:
                        direction = Direction.DOWN
                        return i, direction
            else:
                for i in reversed(search_range[1]):  # position -> 0
                    if call_down[i]:
                        return i, direction
                for i in search_range[1]:  # 0 -> pos
                    if call_up[i]:
                        # if current floor requested follow the request to go up
                        if position == i:
                            direction = Direction.UP
                        return i, direction
                # change direction if requested: down -> up
                for i in search_range[0]:  # pos -> 14
                    if call_up[i]:
                        direction = Direction.UP
                        return i, direction
                for i in reversed(search_range[0]):  # 14 -> pos
                    if call_down[i]:
                        direction = Direction.UP
                        return i, direction

            if searched_one_direction:
                break
            else:
                direction = Direction.UP if direction == Direction.DOWN else Direction.DOWN
                searched_one_direction = True
        return None, direction

    @staticmethod
    def get_next_job(position: int, direction: Direction, next_state: ElevatorState,
                     passengers: list[Person]) -> tuple[int, ElevatorState] | tuple[None, ElevatorState]:
        target_floor: int = Conf.maxFloor - 1 if direction == Direction.UP else 0
        for p in passengers:
            floor = p.schedule[0][1]
            if direction == Direction.UP:
                if floor <= target_floor:
                    next_state = ElevatorState.UP
                    target_floor = floor
            else:
                if floor >= target_floor:
                    next_state = ElevatorState.DOWN
                    target_floor = floor
        if next_state != ElevatorState.WAIT:
            # if passengers with target exist, set new target
            return target_floor, next_state
        else:
            return None, next_state
