# Axon SDK — A simulation framework for spike-timing-based neural computation using the STICK model.
# Copyright (C) 2024–2025 Neucom ApS
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
