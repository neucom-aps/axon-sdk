# Copyright (C) 2025  Neucom Aps
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

from axon_sdk.primitives import (
    SpikingNetworkModule,
    DataEncoder,
)
import math
from typing import Optional


class SignedConstantNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, value: float, module_name: Optional[str] = None) -> None:
        super().__init__(module_name)
        self.encoder = encoder
        self.value = value

        Vt = 10.0
        tm = 100.0
        tf = 20.0
        we = Vt
        Tsyn = 1.0
        f_x = (math.fabs(value) * self.encoder.Tcod) + encoder.Tmin

        # Create constant neuron
        self.recall = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="recall"
        )
        self.output_plus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="output_plus"
        )
        self.output_minus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="output_minus"
        )

        # Connect constant neuron to itself with a delay

        if value >= 0:
            self.connect_neurons(self.recall, self.output_plus, "V", we, Tsyn)
            self.connect_neurons(self.recall, self.output_plus, "V", we, Tsyn + f_x)
        else:
            self.connect_neurons(self.recall, self.output_minus, "V", we, Tsyn)
            self.connect_neurons(self.recall, self.output_minus, "V", we, Tsyn + f_x)


if __name__ == "__main__":
    from axon_sdk import Simulator

    encoder = DataEncoder()
    value = -1.0  # Constant value between 0–1

    constant_network = SignedConstantNetwork(encoder, value)
    sim = Simulator(constant_network, encoder)
    sim.apply_input_spike(constant_network.recall, t=0)
    sim.simulate(simulation_time=120)
    output_spikes = sim.spike_log[constant_network.output_minus.uid]
    print(f"Input value: {value}")
    print(f"Output spikes: {output_spikes}")
