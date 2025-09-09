class DataEncoder:
    def __init__(self, Tmin=10.0, Tcod=100.0):
        self.Tmin = Tmin
        self.Tcod = Tcod
        self.Tmax = Tmin + Tcod

    def encode_value(self, value: float) -> tuple[float, float]:
        assert value >= 0 and value <= 1
        interval = self.Tmin + value * self.Tcod
        return (0, interval)

    def decode_interval(self, spiking_interval: float) -> float:
        value = (spiking_interval - self.Tmin) / self.Tcod
        return value
