class DataEncoder:
    def __init__(self, Tmin=10.0, Tcod=100.0):
        """
        Initialize the encoder with the given minimum interval and coding time.

        Parameters:
        Tmin (float): Minimum spike interval (ms).
        Tcod (float): Interval duration representing the maximum encoded value (ms).
        """
        self.Tmin = Tmin
        self.Tcod = Tcod
        self.Tmax = Tmin + Tcod

    def encode_value(self, value: float) -> tuple[float, float]:
        """
        Encode a value into spike times.

        Parameters:
        value (float): The value to encode, expected between 0 and 1.

        Returns:
        tuple: Two spike times representing the encoded value.
        """
        assert value >= 0 and value <= 1
        interval = self.Tmin + value * self.Tcod
        return (0, interval)

    def decode_interval(self, spiking_interval: float) -> float:
        """
        Decode a spikes interval into a value

        Parameters:
        spiking_interval (float): The value to encode, expected between 0 and 1.

        Returns:
        float: The decoded value
        """
        value = (spiking_interval - self.Tmin) / self.Tcod
        return value
