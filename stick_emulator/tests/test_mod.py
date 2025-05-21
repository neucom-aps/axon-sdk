import pytest

from stick_emulator.networks import ModNetwork
from stick_emulator.primitives import DataEncoder
from stick_emulator.simulator import Simulator


@pytest.mark.parametrize(
    "val1, val2",
    [
        (0.5, 0.5),
        (1.0, 0.13),
        (0.1, 1.0),
        (1.0, 0.5),
        (0.5, 0.45),
        (0.9, 0.3),
        (0.11, 0.11),
        (0.99, 0.01),
        (0.33, 0.12),
        (0.42, 0.36),
    ],
)
def test_mul(val1, val2):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)

    net = ModNetwork(encoder, val2)
    sim = Simulator(net, encoder)
    sim.apply_input_value(val1, neuron=net.input, t0=0)
    sim.simulate(400)

    spikes = sim.spike_log.get(net.output.uid, [])

    assert len(spikes) == 2, f"Expected 2 spikes, got {len(spikes)}"

    interval = spikes[1] - spikes[0]
    decoded_value = encoder.decode_interval(interval)
    actual_value = val1 % val2

    assert actual_value == pytest.approx(
        decoded_value, abs=1e-2
    ), f"Expected decoded value {actual_value}, got {decoded_value}"
