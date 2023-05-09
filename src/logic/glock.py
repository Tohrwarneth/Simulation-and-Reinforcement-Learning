class Glock:
    time_in_sec: float = 0
    tact: int = 0
    end_of_day: bool = False
    peak_times = [(8 * 60, 1), (12 * 60, 1), (17 * 60, 1)]

    @classmethod
    def get_time(cls):
        sec = int(cls.time_in_sec)
        h: int = sec // 60  # sim h = RL m
        m: int = sec % 60  # sim m = RL s
        return h, m

    @classmethod
    def add_delta(cls, delta_time: float):
        cls.time_in_sec += delta_time
        cls.tact += int(delta_time)
        h, m = cls.get_time()
        if h >= 24:
            cls.end_of_day = True
