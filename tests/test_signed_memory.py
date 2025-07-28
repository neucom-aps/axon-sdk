import pytest
from axon_sdk.networks.memory.signed_memory import SignedMemoryNetwork
from axon_sdk.primitives import DataEncoder
from axon_sdk import Simulator, PredSimulator


def encode_and_recall(input_value, is_positive, encoder):
    net = SignedMemoryNetwork(encoder)
    sim = Simulator(net, encoder, dt=0.01)

    if is_positive:
        sim.apply_input_value(abs(input_value), net.input_pos, t0=0)
    else:
        sim.apply_input_value(abs(input_value), net.input_neg, t0=0)

    sim.apply_input_spike(net.recall, t=200)
    sim.simulate(simulation_time=800)

    output_spikes = (
        sim.spike_log.get(net.output_pos.uid, [])
        if is_positive
        else sim.spike_log.get(net.output_neg.uid, [])
    )
    return output_spikes


@pytest.mark.parametrize(
    "input_value, expected_interval",
    [
        (0.2, 30),  # Tmin + (0.2)*Tcod = 30
        (0.5, 60),  # Tmin + (0.5)*Tcod = 60
        (0.7, 80),  # Tmin + (0.7)*Tcod = 80
        (1.0, 110),  # Tmin + (1.0)*Tcod = 110
    ],
)
def test_positive_encode_recall(input_value, expected_interval):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    out_spikes_pos = encode_and_recall(input_value, True, encoder)
    assert len(out_spikes_pos) == 2
    decoded_value_pos = encoder.decode_interval(out_spikes_pos[1] - out_spikes_pos[0])
    expected_value_pos = input_value
    assert expected_value_pos == pytest.approx(decoded_value_pos, abs=1e-2)


@pytest.mark.parametrize(
    "input_value, expected_interval",
    [
        (0.2, 30),  # Tmin + (0.2)*Tcod = 30
        (0.5, 60),  # Tmin + (0.5)*Tcod = 60
        (0.7, 80),  # Tmin + (0.7)*Tcod = 80
        (1.0, 110),  # Tmin + (1.0)*Tcod = 110
    ],
)
def test_negative_encode_recall(input_value, expected_interval):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    out_spikes_neg = encode_and_recall(-input_value, False, encoder)
    assert len(out_spikes_neg) == 2
    decoded_value_neg = encoder.decode_interval(out_spikes_neg[1] - out_spikes_neg[0])
    decoded_value_neg *= -1  # Decoded value neg has to be flipped of its sign manually
    expected_value_neg = -input_value
    assert expected_value_neg == pytest.approx(decoded_value_neg, abs=1e-2)


def test_no_spikes_for_inval_input():
    encoder = DataEncoder()
    net = SignedMemoryNetwork(encoder)
    sim = Simulator(net, encoder, dt=0.01)
    with pytest.raises(ValueError):
        sim.apply_input_value(value=1.5, neuron=net.input_pos, t0=0.0)


@pytest.mark.parametrize(
    "Tmin, Tcod, input_value, expected_interval",
    [
        (5, 50, 0.5, 30),  # Tmin=5, Tcod=50: 5 + (0.5)*50 = 30
        (20, 200, 0.25, 70),  # Tmin=20, Tcod=200: 20 + (0.25)*200 = 70
        (10, 500, 0.75, 385),  # Tmin=10, Tcod=500: 10 + (0.75)*500 = 385
    ],
)
def test_positive_custom_encoder_parameters(Tmin, Tcod, input_value, expected_interval):
    encoder = DataEncoder(Tmin=Tmin, Tcod=Tcod)
    out_spikes_pos = encode_and_recall(input_value, True, encoder)
    expected_value_pos = input_value
    decoded_value_pos = encoder.decode_interval(out_spikes_pos[1] - out_spikes_pos[0])
    assert len(out_spikes_pos) == 2
    assert expected_value_pos == pytest.approx(decoded_value_pos, abs=1e-2)


@pytest.mark.parametrize(
    "Tmin, Tcod, input_value, expected_interval",
    [
        (5, 50, 0.5, 30),  # Tmin=5, Tcod=50: 5 + (0.5)*50 = 30
        (20, 200, 0.25, 70),  # Tmin=20, Tcod=200: 20 + (0.25)*200 = 70
        (10, 500, 0.75, 385),  # Tmin=10, Tcod=500: 10 + (0.75)*500 = 385
    ],
)
def test_negative_custom_encoder_parameters(Tmin, Tcod, input_value, expected_interval):
    encoder = DataEncoder(Tmin=Tmin, Tcod=Tcod)
    out_spikes_neg = encode_and_recall(-input_value, False, encoder)
    expected_value_neg = -input_value
    decoded_value_neg = encoder.decode_interval(out_spikes_neg[1] - out_spikes_neg[0])
    decoded_value_neg *= -1  # Decoded value neg has to be flipped of its sign manually
    assert len(out_spikes_neg) == 2
    assert expected_value_neg == pytest.approx(decoded_value_neg, abs=1e-2)


# ------------------ With predictive simulator ------------------


def pred_encode_and_recall(input_value, is_positive, encoder):
    net = SignedMemoryNetwork(encoder)
    sim = PredSimulator(net, encoder, dt=0.1)

    if is_positive:
        sim.apply_input_value(abs(input_value), net.input_pos, t0=0)
    else:
        sim.apply_input_value(abs(input_value), net.input_neg, t0=0)

    sim.apply_input_spike(net.recall, t=200)
    sim.simulate()

    output_spikes = (
        sim.spike_log.get(net.output_pos.uid, [])
        if is_positive
        else sim.spike_log.get(net.output_neg.uid, [])
    )
    return output_spikes


@pytest.mark.parametrize(
    "input_value, expected_interval",
    [
        (0.2, 30),  # Tmin + (0.2)*Tcod = 30
        (0.5, 60),  # Tmin + (0.5)*Tcod = 60
        (0.7, 80),  # Tmin + (0.7)*Tcod = 80
        (1.0, 110),  # Tmin + (1.0)*Tcod = 110
    ],
)
def test_pred_positive_encode_recall(input_value, expected_interval):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    out_spikes_pos = pred_encode_and_recall(input_value, True, encoder)
    assert len(out_spikes_pos) == 2
    decoded_value_pos = encoder.decode_interval(out_spikes_pos[1] - out_spikes_pos[0])
    expected_value_pos = input_value
    assert expected_value_pos == pytest.approx(decoded_value_pos, abs=1e-2)


@pytest.mark.parametrize(
    "input_value, expected_interval",
    [
        (0.2, 30),  # Tmin + (0.2)*Tcod = 30
        (0.5, 60),  # Tmin + (0.5)*Tcod = 60
        (0.7, 80),  # Tmin + (0.7)*Tcod = 80
        (1.0, 110),  # Tmin + (1.0)*Tcod = 110
    ],
)
def test_pred_negative_encode_recall(input_value, expected_interval):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    out_spikes_neg = pred_encode_and_recall(-input_value, False, encoder)
    assert len(out_spikes_neg) == 2
    decoded_value_neg = encoder.decode_interval(out_spikes_neg[1] - out_spikes_neg[0])
    decoded_value_neg *= -1  # Decoded value neg has to be flipped of its sign manually
    expected_value_neg = -input_value
    assert expected_value_neg == pytest.approx(decoded_value_neg, abs=1e-2)


def test_pred_no_spikes_for_inval_input():
    encoder = DataEncoder()
    net = SignedMemoryNetwork(encoder)
    sim = PredSimulator(net, encoder, dt=0.01)
    with pytest.raises(ValueError):
        sim.apply_input_value(value=1.5, neuron=net.input_pos, t0=0.0)


@pytest.mark.parametrize(
    "Tmin, Tcod, input_value, expected_interval",
    [
        (5, 50, 0.5, 30),  # Tmin=5, Tcod=50: 5 + (0.5)*50 = 30
        (20, 200, 0.25, 70),  # Tmin=20, Tcod=200: 20 + (0.25)*200 = 70
        (10, 500, 0.75, 385),  # Tmin=10, Tcod=500: 10 + (0.75)*500 = 385
    ],
)
def test_pred_positive_custom_encoder_parameters(
    Tmin, Tcod, input_value, expected_interval
):
    encoder = DataEncoder(Tmin=Tmin, Tcod=Tcod)
    out_spikes_pos = pred_encode_and_recall(input_value, True, encoder)
    expected_value_pos = input_value
    decoded_value_pos = encoder.decode_interval(out_spikes_pos[1] - out_spikes_pos[0])
    assert len(out_spikes_pos) == 2


@pytest.mark.parametrize(
    "Tmin, Tcod, input_value, expected_interval",
    [
        (5, 50, 0.5, 30),  # Tmin=5, Tcod=50: 5 + (0.5)*50 = 30
        (20, 200, 0.25, 70),  # Tmin=20, Tcod=200: 20 + (0.25)*200 = 70
        (10, 500, 0.75, 385),  # Tmin=10, Tcod=500: 10 + (0.75)*500 = 385
    ],
)
def test_pred_negative_custom_encoder_parameters(
    Tmin, Tcod, input_value, expected_interval
):
    encoder = DataEncoder(Tmin=Tmin, Tcod=Tcod)
    out_spikes_neg = pred_encode_and_recall(-input_value, False, encoder)
    expected_value_neg = -input_value
    decoded_value_neg = encoder.decode_interval(out_spikes_neg[1] - out_spikes_neg[0])
    decoded_value_neg *= -1  # Decoded value neg has to be flipped of its sign manually
    assert len(out_spikes_neg) == 2
    assert expected_value_neg == pytest.approx(decoded_value_neg, abs=1e-2)
