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


class ExponentialNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, module_name: Optional[str] = None) -> None:
        super().__init__(module_name)
        self.encoder = encoder

        # Parameters
        Vt = 10.0
        tm = 100.0
        self.tf = 20.0
        Tsyn = 1.0
        Tmin = encoder.Tmin

        we = Vt
        wi = -Vt
        gmult = (Vt * tm) / self.tf
        wacc_bar = Vt * tm / encoder.Tcod  # To ensure V = Vt at Tcod

        # Neurons
        self.input = self.add_neuron(Vt, tm, self.tf, neuron_name="input")
        self.first = self.add_neuron(Vt, tm, self.tf, neuron_name="first")
        self.last = self.add_neuron(Vt, tm, self.tf, neuron_name="last")
        self.acc = self.add_neuron(Vt, tm, self.tf, neuron_name="acc")
        self.output = self.add_neuron(Vt, tm, self.tf, neuron_name="output")

        # Connections from input neuron
        self.connect_neurons(self.input, self.first, "V", we, Tsyn)
        self.connect_neurons(self.input, self.last, "V", 0.5 * we, Tsyn)

        # Inhibit first after spike
        self.connect_neurons(self.first, self.first, "V", wi, Tsyn)

        # Exponential computation:
        # 1. First spike → apply gf with delay = Tsyn + Tmin
        self.connect_neurons(self.first, self.acc, "gf", gmult, Tsyn + Tmin)
        self.connect_neurons(self.first, self.acc, "gate", 1, Tsyn + Tmin)
        # 2. Last spike → open gate
        self.connect_neurons(self.last, self.acc, "gate", -1.0, Tsyn)

        # 3. Last spike → add ge to trigger spike after ts
        self.connect_neurons(self.last, self.acc, "ge", wacc_bar, Tsyn)

        # Readout to output
        self.connect_neurons(self.acc, self.output, "V", we, Tsyn + Tmin)
        self.connect_neurons(self.last, self.output, "V", we, 2 * Tsyn)


def expected_exp_output_delay(x, encoder: DataEncoder, tf):
    try:
        delay = encoder.Tcod * math.exp(-x * encoder.Tcod / tf)
        Tout = encoder.Tmin + delay
        return Tout
    except:
        return float("nan")


def decode_exponential(output_interval, encoder: DataEncoder, tf):
    return ((output_interval - encoder.Tmin) / encoder.Tcod) ** (-tf / encoder.Tcod)


if __name__ == "__main__":
    from axon_sdk import Simulator

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = ExponentialNetwork(encoder)
    sim = Simulator(net, encoder, dt=0.01)

    value = 0.5
    sim.apply_input_value(value, neuron=net.input, t0=10)
    sim.simulate(150)

    output_spikes = sim.spike_log.get(net.output.uid, [])
    if len(output_spikes) == 2:
        out_interval = output_spikes[1] - output_spikes[0]
        print(f"Input value: {value}")
        print(
            f"Expected exp value: {math.exp(value)}, decoded exp value {decode_exponential(out_interval, encoder, net.tf)}, "
        )
        print(
            f"Expected exp delay: {expected_exp_output_delay(value, encoder, net.tf):.3f} ms"
        )
        print(f"Measured exp delay: {out_interval:.3f} ms")
    else:
        print(f"Expected 2 output spikes, got {len(output_spikes)}")

    # sim.apply_input_spike(net.input, t=0.0)
    # sim.simulate(300)
