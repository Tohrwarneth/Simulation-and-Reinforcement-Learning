import enums
from re_learner import Net, NetCoder
from re_learner.IReinforcementDecider import IReinforcementDecider


class ReinforcementDecider(IReinforcementDecider):
    net: Net

    @classmethod
    def init(cls, net: Net):
        cls.net = net

    @classmethod
    def get_decision(cls, sim) -> tuple[enums.ElevatorState, enums.ElevatorState, enums.ElevatorState]:
        decisions_float = cls.net.decide_for_action(sim.get_game_state())
        decisions_states = NetCoder.decision_to_states(decisions_float)
        return decisions_states
