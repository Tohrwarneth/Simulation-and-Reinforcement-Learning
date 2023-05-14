class Clock:
    time_in_sec: float = 0
    tact: int = 0
    end_of_day: bool = False
    peak_times = [(7 * 60, 8, 10 * 60), (11 * 60, 12, 13 * 60), (16 * 60, 17, 18 * 60)]

    @classmethod
    def get_time(cls):
        sec = int(cls.time_in_sec)
        h: int = sec // 60  # sim h = RL m
        m: int = sec % 60  # sim m = RL s
        return h, m

    @classmethod
    def add_delta(cls, delta_time: float):
        cls.time_in_sec += delta_time
        cls.tact = int(cls.time_in_sec)
        h, m = cls.get_time()
        if h >= 24:
            cls.end_of_day = True

    @classmethod
    def get_peak(cls) -> int:
        morning, lunch, evening = cls.peak_times
        sec = cls.tact
        if morning[0] < sec < morning[2]:
            return morning[1]
        elif lunch[0] < sec < lunch[2]:
            return lunch[1]
        elif evening[0] < sec < evening[2]:
            return evening[1]
        else:
            return 0
