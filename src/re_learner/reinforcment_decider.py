from re_learner import Net
from re_learner.IReinforcementDecider import IReinforcementDecider


class ReinforcementDecider(IReinforcementDecider):
    net: Net

    @classmethod
    def init(cls, net: Net):
        cls.net = net

    @classmethod
    def get_decision(cls, sim) -> tuple[int, int, int]:
        decisions_float = cls.net.decide_for_action(sim.get_game_state())
        decisions_int = (round(decisions_float[0]), round(decisions_float[1]), round(decisions_float[2]))
        return decisions_int
