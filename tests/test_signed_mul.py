import pytest

from axon_sdk.networks import SignedMultiplierNetwork
from axon_sdk.primitives import DataEncoder
from axon_sdk import Simulator, PredSimulator


@pytest.mark.parametrize(
    "val1, val2",
    [
        (-0.5, 0.5),
        (-1.0, 1.0),
        (-0.1, 1.0),
        (-1.0, 0.5),
        (-0.5, -0.45),
        (0.3, -0.9),
        (0.11, -0.11),
        (0.01, -0.99),
        (-0.12, 0.33),
        (0.42, 0.42),
    ],
)
def test_mul(val1, val2):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)

    net = SignedMultiplierNetwork(encoder)
    sim = Simulator(net, encoder)

    if val1 > 0:
        sim.apply_input_value(abs(val1), neuron=net.input1_plus, t0=0)
    else:
        sim.apply_input_value(abs(val1), neuron=net.input1_minus, t0=0)
    if val2 > 0:
        sim.apply_input_value(abs(val2), neuron=net.input2_plus, t0=0)
    else:
        sim.apply_input_value(abs(val2), neuron=net.input2_minus, t0=0)

    sim.simulate(400)

    spikes_plus = sim.spike_log.get(net.output_plus.uid, [])
    spikes_minus = sim.spike_log.get(net.output_minus.uid, [])

    out_result = val1 * val2
    if out_result > 0:
        assert len(spikes_plus) == 2
        assert len(spikes_minus) == 0
        interval = spikes_plus[1] - spikes_plus[0]
        decoded_val = encoder.decode_interval(interval)
        assert out_result == pytest.approx(decoded_val, abs=1e-2)

    elif out_result < 0:
        assert len(spikes_minus) == 2
        assert len(spikes_plus) == 0
        interval = spikes_minus[1] - spikes_minus[0]
        decoded_val = -1 * encoder.decode_interval(interval)
        assert out_result == pytest.approx(decoded_val, abs=1e-2)


@pytest.mark.parametrize(
    "val1, val2",
    [
        (0.0, 0.0),
    ],
)
def test_mul_zero(val1, val2):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)

    net = SignedMultiplierNetwork(encoder)
    sim = Simulator(net, encoder)
    sim.apply_input_value(abs(val1), neuron=net.input2_plus, t0=0)
    sim.apply_input_value(abs(val2), neuron=net.input2_plus, t0=0)
    sim.simulate(400)

    spikes_plus = sim.spike_log.get(net.output_plus.uid, [])
    spikes_minus = sim.spike_log.get(net.output_minus.uid, [])

    # For inputs of 0, expects no outputs (by design)
    assert len(spikes_plus) == 0
    assert len(spikes_minus) == 0


# ------------------ With predictive simulator ------------------


@pytest.mark.parametrize(
    "val1, val2",
    [
        (-0.5, 0.5),
        (-1.0, 1.0),
        # (-0.1, 1.0),
        # (-1.0, 0.5),
        (-0.5, -0.45),
        (0.3, -0.9),
        (0.11, -0.11),
        (0.01, -0.99),
        (-0.12, 0.33),
        (0.42, 0.42),
    ],
)
def test_pred_mul(val1, val2):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)

    net = SignedMultiplierNetwork(encoder)
    sim = PredSimulator(net, encoder, dt=0.01)

    if val1 > 0:
        sim.apply_input_value(abs(val1), neuron=net.input1_plus, t0=0)
    else:
        sim.apply_input_value(abs(val1), neuron=net.input1_minus, t0=0)
    if val2 > 0:
        sim.apply_input_value(abs(val2), neuron=net.input2_plus, t0=0)
    else:
        sim.apply_input_value(abs(val2), neuron=net.input2_minus, t0=0)

    sim.simulate()

    spikes_plus = sim.spike_log.get(net.output_plus.uid, [])
    spikes_minus = sim.spike_log.get(net.output_minus.uid, [])

    out_result = val1 * val2
    if out_result > 0:
        assert len(spikes_plus) == 2
        assert len(spikes_minus) == 0
        interval = spikes_plus[1] - spikes_plus[0]
        decoded_val = encoder.decode_interval(interval)
        assert out_result == pytest.approx(decoded_val, abs=1e-2)

    elif out_result < 0:
        assert len(spikes_minus) == 2
        assert len(spikes_plus) == 0
        interval = spikes_minus[1] - spikes_minus[0]
        decoded_val = -1 * encoder.decode_interval(interval)
        assert out_result == pytest.approx(decoded_val, abs=1e-2)


@pytest.mark.parametrize(
    "val1, val2",
    [
        (0.0, 0.0),
    ],
)
def test_pred_mul_zero(val1, val2):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)

    net = SignedMultiplierNetwork(encoder)
    sim = PredSimulator(net, encoder, dt=0.01)
    sim.apply_input_value(abs(val1), neuron=net.input2_plus, t0=0)
    sim.apply_input_value(abs(val2), neuron=net.input2_plus, t0=0)
    sim.simulate()

    spikes_plus = sim.spike_log.get(net.output_plus.uid, [])
    spikes_minus = sim.spike_log.get(net.output_minus.uid, [])

    # For inputs of 0, expects no outputs (by design)
    assert len(spikes_plus) == 0
    assert len(spikes_minus) == 0
