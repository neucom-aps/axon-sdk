import pytest
from maximum_network import MaximumNetwork
from primitives import DataEncoder
from simulator import Simulator


# Helper function to run encode-recall simulation
def simulate_min_net(input1_value, input2_value):
    encoder = DataEncoder(Tcod=100)
    maxnet = MaximumNetwork(encoder)

    sim = Simulator(maxnet, encoder, dt=0.01)
    sim.apply_input_value(value=input1_value, neuron=maxnet.input_1, t0=0)
    sim.apply_input_value(value=input2_value, neuron=maxnet.input_2, t0=0)

    # Run simulation for enough time to capture output
    sim.simulate(simulation_time=500)

    output_spike_log = sim.spike_log.get(maxnet.output.uid, [])

    return output_spike_log


@pytest.mark.parametrize(
    "input1_value, input2_value, expected_minimum",
    [
        (0.89, 0.65, 0.89),  # Both values are positive
        (0.12, 0.13, 0.13),  # First value is smaller
        (0.4, 0.3, 0.4),  # Second value is smaller
        (0.5, 0.5, 0.5),  # Values almost equal
        (0.0, 0.1, 0.1),  # One value is zero
        (0.1, 0.0, 0.1),  # One value is zero
        # (-0.2, -0.3, -0.3),  # Both values are negative
        # (-0.5, -0.4, -0.5),  # First value is smaller
        # (-0.6, -0.7, -0.7),  # Second value is smaller
    ],
)
def test_minimum_network_output(input1_value, input2_value, expected_minimum):
    encoder = DataEncoder(Tcod=100)
    output_spikes = simulate_min_net(input1_value, input2_value)

    assert (
        len(output_spikes) == 2
    ), f"Expected exactly two output spikes for inputs {input1_value} and {input2_value}, got {len(output_spikes)}"

    out_val = encoder.decode_interval(output_spikes[1] - output_spikes[0])
    assert expected_minimum == pytest.approx(
        out_val, abs=1e-2
    ), f"Expected minimum value {expected_minimum} for inputs {input1_value} and {input2_value}, got {out_val}"


"""
def test_negative_input():
    input1_value = -0.89
    input2_value = -0.65

    output_spikes = simulate_min_net(input1_value, input2_value)

    assert (
        len(output_spikes) >= 2
    ), f"Expected exactly two output spikes for negative inputs {input1_value} and {input2_value}, got {len(output_spikes)}"

    out_val = (
        input1_value if output_spikes[0] < output_spikes[1] else input2_value
    )
    assert -0.89 == pytest.approx(
        out_val, abs=1e-2
    ), f"Expected minimum value -0.89 for negative inputs {input1_value} and {input2_value}, got {out_val}"
"""


def test_zero_input():
    input1_value = 0.0
    input2_value = 0.5

    encoder = DataEncoder(Tcod=100)
    output_spikes = simulate_min_net(input1_value, input2_value)

    assert (
        len(output_spikes) == 2
    ), f"Expected exactly two output spikes for inputs {input1_value} and {input2_value}, got {len(output_spikes)}"

    out_val = encoder.decode_interval(output_spikes[1] - output_spikes[0])
    assert 0.5 == pytest.approx(
        out_val, abs=1e-2
    ), f"Expected minimum value 0.0 for inputs {input1_value} and {input2_value}, got {out_val}"


def test_identical_inputs():
    input1_value = 0.5
    input2_value = 0.5

    encoder = DataEncoder(Tcod=100)
    output_spikes = simulate_min_net(input1_value, input2_value)

    assert (
        len(output_spikes) == 2
    ), f"Expected exactly two output spikes for identical inputs {input1_value} and {input2_value}, got {len(output_spikes)}"

    out_val = encoder.decode_interval(output_spikes[1] - output_spikes[0])
    assert 0.5 == pytest.approx(
        out_val, abs=1e-2
    ), f"Expected minimum value 0.5 for identical inputs {input1_value} and {input2_value}, got {out_val}"
