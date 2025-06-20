
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

import pytest

from axon_sdk.networks import DivNetwork
from axon_sdk.primitives import DataEncoder
from axon_sdk.simulator import Simulator


# x1 <= x2 (x1 must be smaller or equal than x2)
@pytest.mark.parametrize(
    "x1, x2",
    [
        (1.0, 1.0),
        (0.5, 0.5),
        (0.5, 1.0),
        (0.2, 0.2),
        (0.2, 0.2),
        (0.82, 0.91),
        (0.52, 0.91),
        (0.52, 0.63),
        (0.22, 0.91),
        (0.22, 0.63),
        (0.22, 0.31),
        (0.01, 0.91),
        (0.01, 0.63),
        (0.01, 0.63),
    ],
)
def test_div_2_decimals(x1, x2):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)

    assert x1 <= x2, f"x1<=x2, but got x1: {x1} and x2: {x2}"

    net = DivNetwork(encoder)
    sim = Simulator(net, encoder)
    sim.apply_input_value(value=x1, neuron=net.input1, t0=0)
    sim.apply_input_value(value=x2, neuron=net.input2, t0=0)
    sim.simulate(300)

    spikes = sim.spike_log.get(net.output.uid, [])

    assert len(spikes) == 2, f"Expected 2 spikes; got {len(spikes)}"

    expected_value = x1 / x2
    decoded_value = encoder.decode_interval(spikes[1] - spikes[0])

    assert expected_value == pytest.approx(decoded_value, abs=1e-4), f"Expected value {expected_value}, got {decoded_value}"


# x1 <= x2 (x1 must be smaller or equal than x2)
@pytest.mark.parametrize(
    "x1, x2",
    [
        (0.01, 0.1),
        (0.001, 0.1),
        (0.0001, 0.1),
        (0.01, 0.01),
        (0.001, 0.01),
        (0.0001, 0.01),
        (0.001, 0.001),
        (0.0001, 0.001),
    ],
)
def test_div_4_decimals(x1, x2):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)

    assert x1 <= x2, f"x1<=x2, but got x1: {x1} and x2: {x2}"

    net = DivNetwork(encoder)
    sim = Simulator(net, encoder)
    sim.apply_input_value(value=x1, neuron=net.input1, t0=0)
    sim.apply_input_value(value=x2, neuron=net.input2, t0=0)
    sim.simulate(300)

    spikes = sim.spike_log.get(net.output.uid, [])

    assert len(spikes) == 2, f"Expected 2 spikes; got {len(spikes)}"

    expected_value = x1 / x2
    decoded_value = encoder.decode_interval(spikes[1] - spikes[0])

    assert expected_value == pytest.approx(decoded_value, abs=1e-4), f"Expected value {expected_value}, got {decoded_value}"
