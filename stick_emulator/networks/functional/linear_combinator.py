from stick_emulator.primitives import (
    ExplicitNeuron,
    EventQueue,
    SpikingNetworkModule,
    DataEncoder,
)
from stick_emulator.networks import SubtractorNetwork, SynchronizerNetwork
import math


class LinearCombinatorNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, N: int, coeff: list, prefix: str = ""):
        super().__init__()
        self.encoder = encoder

        # Constants
        Vt = 10.0
        tm = 100.0
        tf = 20.0  ## Increasing this makes Log more accurate
        Tsyn = 1.0
        Tmin = encoder.Tmin

        we = Vt
        wi = -Vt
        gmult = (Vt * tm) / tf
        wacc = (Vt * tm) / encoder.Tcod

        self.acc1_plus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "_acc1_plus"
        )
        self.acc1_minus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "_acc1_minus"
        )
        self.acc2_plus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "_acc2_plus"
        )
        self.acc2_minus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "_acc2_minus"
        )
        self.inter_minus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "_inter_minus"
        )
        self.inter_plus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "_inter_plus"
        )
        self.output_plus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "_output_plus"
        )
        self.output_minus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "_output_minus"
        )
        self.start = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "_start")
        self.sync = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "_sync")

        self.sync_network = SynchronizerNetwork(encoder, 2)
        self.subtractor_network = SubtractorNetwork(encoder)

        self.add_subnetwork(self.sync_network)
        self.add_subnetwork(self.subtractor_network)

        # Connect outputs of sync to inputs of subtractor
        self.connect_neurons(
            self.sync_network.output_neurons[0],
            self.subtractor_network.input1,
            "V",
            we,
            Tsyn,
        )
        self.connect_neurons(
            self.sync_network.output_neurons[1],
            self.subtractor_network.input2,
            "V",
            we,
            Tsyn,
        )
        # connect outputs (plus/minus) to start and output +/- neurons
        self.connect_neurons(
            self.subtractor_network.output_plus, self.start, "V", we, Tsyn
        )
        self.connect_neurons(
            self.subtractor_network.output_minus, self.start, "V", we, Tsyn
        )
        self.connect_neurons(
            self.subtractor_network.output_plus, self.output_plus, "V", we, Tsyn
        )
        self.connect_neurons(
            self.subtractor_network.output_minus,
            self.output_minus,
            "V",
            we,
            Tsyn,
        )
        # Recurrent connection to start
        self.connect_neurons(self.start, self.start, "V", wi, Tsyn)

        # connect Inter+ to input0 of synchronizer
        self.connect_neurons(
            self.inter_plus, self.sync_network.input_neurons[0], "V", we, Tsyn
        )
        # connect Inter- to input1 of synchronizer
        self.connect_neurons(
            self.inter_minus, self.sync_network.input_neurons[1], "V", we, Tsyn
        )

        # Connect sync to acc1/2 + and minus with ge synapse and wacc
        self.connect_neurons(self.sync, self.acc1_plus, "ge", wacc, Tsyn)
        self.connect_neurons(self.sync, self.acc1_minus, "ge", wacc, Tsyn)
        self.connect_neurons(self.sync, self.acc2_plus, "ge", wacc, Tsyn)
        self.connect_neurons(self.sync, self.acc2_minus, "ge", wacc, Tsyn)

        # Connect acc1+ to inter+ with V synapse and Tsyn
        self.connect_neurons(self.acc1_plus, self.inter_plus, "V", we, Tsyn)
        # Connect acc1- to inter- with V synapse and Tsyn
        self.connect_neurons(self.acc1_minus, self.inter_minus, "V", we, Tsyn)
        # Connect acc2+ to inter+ with V synapse and Tsyn + Tmin
        self.connect_neurons(self.acc2_plus, self.inter_plus, "V", we, Tsyn + Tmin)
        # Connect acc2- to inter- with V synapse and Tsyn + Tmin
        self.connect_neurons(self.acc2_minus, self.inter_minus, "V", we, Tsyn + Tmin)

        self.input_neurons = []

        for i in range(N):
            # Get the coefficient as absolute value
            a_i = coeff[i]
            a_i = abs(a_i)
            # Create input +/- neurons
            input_plus = ExplicitNeuron(
                Vt=Vt, tm=tm, tf=tf, neuron_name=f"input_plus_{i}"
            )
            input_minus = ExplicitNeuron(
                Vt=Vt, tm=tm, tf=tf, neuron_name=f"input_minus_{i}"
            )
            # Create first and last +/-
            first_plus = ExplicitNeuron(
                Vt=Vt, tm=tm, tf=tf, neuron_name=f"first_plus_{i}"
            )
            first_minus = ExplicitNeuron(
                Vt=Vt, tm=tm, tf=tf, neuron_name=f"first_minus_{i}"
            )
            last_plus = ExplicitNeuron(
                Vt=Vt, tm=tm, tf=tf, neuron_name=f"last_plus_{i}"
            )
            last_minus = ExplicitNeuron(
                Vt=Vt, tm=tm, tf=tf, neuron_name=f"last_minus_{i}"
            )

            self.input_neurons.append(input_plus)
            self.input_neurons.append(input_minus)

            # Connect we (V) to from input + to first +
            self.connect_neurons(input_plus, first_plus, "V", we, Tsyn)
            self.connect_neurons(input_plus, last_plus, "V", 0.5 * we, Tsyn)

            # Same for negative
            self.connect_neurons(input_minus, first_minus, "V", we, Tsyn)
            self.connect_neurons(input_minus, last_minus, "V", 0.5 * we, Tsyn)

            # recurrent connection for first plus and minus
            self.connect_neurons(first_plus, first_plus, "V", wi, Tsyn)
            self.connect_neurons(first_minus, first_minus, "V", wi, Tsyn)

            self.connect_neurons(last_plus, self.sync, "V", we / N, Tsyn)
            self.connect_neurons(last_minus, self.sync, "V", we / N, Tsyn)

            # Connect first_plus with ge to acc1_plus where weight is |a_i|*wacc
            self.connect_neurons(
                first_plus, self.acc1_plus, "ge", a_i * wacc, Tsyn + Tmin
            )
            self.connect_neurons(last_plus, self.acc1_plus, "ge", -a_i * wacc, Tsyn)

            # Connect first_minus with ge to acc1_minus where weight is |a_i|*wacc
            self.connect_neurons(
                first_minus, self.acc1_minus, "ge", a_i * wacc, Tsyn + Tmin
            )
            self.connect_neurons(last_minus, self.acc1_minus, "ge", -a_i * wacc, Tsyn)


def decode_spike_interval(spikes, encoder):
    if len(spikes) < 2:
        return None
    interval = spikes[1] - spikes[0]
    return encoder.decode_interval(interval)


def run_test(coeffs, inputs, Tmin=10.0, Tcod=100.0):
    encoder = DataEncoder(Tmin=Tmin, Tcod=Tcod)
    N = len(coeffs)
    net = LinearCombinatorNetwork(encoder, N=N, coeff=coeffs)

    sim = Simulator(net, encoder, dt=0.01)

    # Apply inputs
    """
    for i, val in enumerate(inputs):
        if val >= 0:
            sim.apply_input_value(val, net.input_neurons[2 * i], t0=0)
        else:
            sim.apply_input_value(-val, net.input_neurons[2 * i + 1], t0=0)
            print(net.input_neurons[2 * i + 1])"
    """
    for i, val in enumerate(inputs):
        product = coeffs[i] * inputs[i]
        if product >= 0:
            sim.apply_input_value(abs(inputs[i]), net.input_neurons[2 * i], t0=0)
        else:
            sim.apply_input_value(abs(inputs[i]), net.input_neurons[2 * i + 1], t0=0)

    sim.simulate(500)

    print("\n==========================")
    print(f"Test case: inputs={inputs}, coeffs={coeffs}")
    expected = sum(c * x for c, x in zip(coeffs, inputs))
    print(f"✅ Expected linear combination: {expected:.3f}")

    plus_spikes = sim.spike_log.get(net.output_plus.uid, [])
    minus_spikes = sim.spike_log.get(net.output_minus.uid, [])

    decoded_plus = decode_spike_interval(plus_spikes, encoder)
    decoded_minus = decode_spike_interval(minus_spikes, encoder)

    if decoded_plus is not None:
        print(
            f"🟢 output+ interval = {plus_spikes[1] - plus_spikes[0]:.3f} ms → decoded = {decoded_plus:.3f}"
        )
    else:
        print("🔴 output+ did not spike twice.")

    if decoded_minus is not None:
        print(
            f"🟢 output- interval = {minus_spikes[1] - minus_spikes[0]:.3f} ms → decoded = -{decoded_minus:.3f}"
        )
    else:
        print("🔴 output- did not spike twice.")

    result = 0
    if decoded_plus is not None:
        result += decoded_plus
    if decoded_minus is not None:
        result -= decoded_minus

    print(f"🔍 Reconstructed value from spikes: {result:.3f}")
    print("==========================\n")
    return result


if __name__ == "__main__":
    # Run a sweep of test cases
    test_cases = [
        ([-0.5, -0.5], [1.0, 1.0]),
        ([0.5, -0.5], [0.5, 1.0]),
        ([-1.0, -1.0], [-0.8, 0.3]),
        ([-1.0, -0.25], [-0.6, -0.4]),
        ([0.5, 0.5], [-0.7, -0.2]),
        ([0.5, 0.5], [-0.7, -0.2]),
    ]

    for coeffs, inputs in test_cases:
        run_test(coeffs, inputs)
