import pytest
from stick_emulator.networks import ExponentialNetwork
from stick_emulator.primitives import DataEncoder
from stick_emulator import Simulator

T_F = 50.0

"""
NOTE: test values computed using

def expected_exp_output_delay(x, Tmin=10.0, Tcod=100.0, tf=T_F):
    Tcod_val = x * Tcod
    try:
        delay = Tcod * math.exp(-Tcod_val / tf)
        Tout = Tmin + delay
        return Tout
    except:
        return float("nan")

"""


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        (0.1, 91.87307530779819),  # Expected delay for log(0.1)
        (0.5, 46.787944117144235),  # Expected delay for log(0.5)
        (0.9, 26.529888822158654),  # Expected delay for log(0.9)
    ],
)
def test_log_output_delay(input_value, expected_output):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = ExponentialNetwork(encoder)

    sim = Simulator(net, encoder, dt=0.01)
    sim.apply_input_value(input_value, neuron=net.input, t0=0)
    sim.simulate(300)

    output_spikes = sim.spike_log.get(net.output.uid, [])

    assert len(output_spikes) == 2
    output_interval = output_spikes[1] - output_spikes[0]

    expected_output_value = encoder.decode_interval(expected_output)
    decoded_value = encoder.decode_interval(output_interval)

    assert pytest.approx(decoded_value, abs=1e-2) == expected_output_value


@pytest.mark.parametrize(
    "Tmin, Tcod, input_value, expected_output",
    [
        (
            5,
            50,
            0.25,
            43.94003915357025,
        ),  # Expected delay for log(0.25) with custom encoder parameters
        (
            20,
            200,
            0.1,
            154.06400920712787,
        ),  # Expected delay for log(0.1) with custom encoder parameters
        (
            10,
            500,
            0.9,
            10.06170490204334,
        ),  # Expected delay for log(0.9) with custom encoder parameters
    ],
)
def test_custom_encoder_parameters(Tmin, Tcod, input_value, expected_output):
    custom_encoder = DataEncoder(Tmin=Tmin, Tcod=Tcod)
    net = ExponentialNetwork(custom_encoder)

    sim = Simulator(net, custom_encoder, dt=0.01)
    sim.apply_input_value(input_value, neuron=net.input, t0=0)
    sim.simulate(600)

    output_spikes = sim.spike_log.get(net.output.uid, [])

    assert len(output_spikes) == 2
    output_interval = output_spikes[1] - output_spikes[0]

    expected_output_value = custom_encoder.decode_interval(expected_output)
    decoded_value = custom_encoder.decode_interval(output_interval)

    assert pytest.approx(decoded_value, abs=1e-2) == expected_output_value
