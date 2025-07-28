import pytest

from axon_sdk.networks import LinearCombinatorNetwork
from axon_sdk.primitives import DataEncoder
from axon_sdk import Simulator, PredSimulator


@pytest.mark.parametrize(
    "coeffs, inputs",
    [
        ([1.0, 1.0, 1.0], [0.1, 0.1, 0.1]),
        ([0.5, 0.5, 0.5], [0.1, 0.1, 0.1]),
        ([0.9, 0.9, 0.9], [0.3, 0.3, 0.3]),
        ([-0.9, 0.9, -0.9], [0.3, 0.3, 0.3]),
        ([-0.5, -0.5, 0.5], [0.1, 0.1, 0.1]),
        ([0.5], [0.1]),
        ([-0.5], [0.1]),
        ([-0.5], [-0.1]),
    ],
)
def test_linmul(coeffs: list[float], inputs: list[float]):
    assert len(coeffs) == len(inputs)

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = LinearCombinatorNetwork(encoder, N=len(coeffs), coeff=coeffs)
    sim = Simulator(net, encoder)

    for idx, inp_val in enumerate(inputs):
        if inp_val >= 0:
            sim.apply_input_value(abs(inp_val), net.input_plus[idx], t0=0)
        else:
            sim.apply_input_value(abs(inp_val), net.input_minus[idx], t0=0)

    sim.simulate(400)

    spikes_plus = sim.spike_log.get(net.output_plus.uid, [])
    spikes_minus = sim.spike_log.get(net.output_minus.uid, [])

    actual_result = sum(c * i for c, i in zip(coeffs, inputs))
    decoded_result = None

    if actual_result > 0:
        assert len(spikes_plus) == 2
        assert len(spikes_minus) == 0
        interval = spikes_plus[1] - spikes_plus[0]
        decoded_result = encoder.decode_interval(interval)

    elif actual_result < 0:
        assert len(spikes_minus) == 2
        assert len(spikes_plus) == 0
        interval = spikes_minus[1] - spikes_minus[0]
        decoded_result = -1 * encoder.decode_interval(interval)

    assert actual_result == pytest.approx(decoded_result, abs=1e-2)


# ------------------ With predictive simulator ------------------


@pytest.mark.parametrize(
    "coeffs, inputs",
    [
        ([1.0, 1.0, 1.0], [0.1, 0.1, 0.1]),
        ([0.5, 0.5, 0.5], [0.1, 0.1, 0.1]),
        ([0.9, 0.9, 0.9], [0.3, 0.3, 0.3]),
        ([-0.9, 0.9, -0.9], [0.3, 0.3, 0.3]),
        ([-0.5, -0.5, 0.5], [0.1, 0.1, 0.1]),
        ([0.5], [0.1]),
        ([-0.5], [0.1]),
        ([-0.5], [-0.1]),

    ],
)
def test_pred_linmul(coeffs: list[float], inputs: list[float]):
    assert len(coeffs) == len(inputs)
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = LinearCombinatorNetwork(encoder, N=len(coeffs), coeff=coeffs)
    sim = PredSimulator(net, encoder, dt=0.01)

    for idx, inp_val in enumerate(inputs):
        if inp_val >= 0:
            sim.apply_input_value(abs(inp_val), net.input_plus[idx], t0=0)
        else:
            sim.apply_input_value(abs(inp_val), net.input_minus[idx], t0=0)

    sim.simulate()

    spikes_plus = sim.spike_log.get(net.output_plus.uid, [])
    spikes_minus = sim.spike_log.get(net.output_minus.uid, [])

    actual_result = sum(c * i for c, i in zip(coeffs, inputs))
    decoded_result = None

    if actual_result > 0:
        assert len(spikes_plus) == 2
        assert len(spikes_minus) == 0
        interval = spikes_plus[1] - spikes_plus[0]
        decoded_result = encoder.decode_interval(interval)

    elif actual_result < 0:
        assert len(spikes_minus) == 2
        assert len(spikes_plus) == 0
        interval = spikes_minus[1] - spikes_minus[0]
        decoded_result = -1 * encoder.decode_interval(interval)

    assert actual_result == pytest.approx(decoded_result, abs=1e-2)