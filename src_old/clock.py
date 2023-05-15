import simpy


class Clock:
    tact: int = 0
    delta_time = 0
    end_of_day: bool = False
    peak_times = [(8 * 60, 30), (12 * 60, 1), (16 * 60, 1)]
    pause_time = 30
    env: simpy.Environment

    @classmethod
    def get_time(cls):
        sec = cls.tact
        h: int = sec // 60  # sim h = RL m
        m: int = sec % 60  # sim m = RL s
        return h, m

    @classmethod
    def add_delta(cls, delta_time: float):
        cls.delta_time = delta_time
        cls.tact = int(cls.env.now)
        h, m = cls.get_time()
        if h >= 24:
            cls.end_of_day = True

    @classmethod
    def get_peak(cls) -> tuple[int, int]:
        # morning, lunch, evening = cls.peak_times
        return cls.peak_times
        # sec = cls.tact
        # if morning[0] < sec < morning[2]:
        #     return morning[1]
        # elif lunch[0] < sec < lunch[2]:
        #     return lunch[1]
        # elif evening[0] < sec < evening[2]:
        #     return evening[1]
        # else:
        #     return 0
