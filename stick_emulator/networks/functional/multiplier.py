from stick_emulator.primitives import (
    SpikingNetworkModule,
    ExplicitNeuron,
    DataEncoder,
)


class MultiplierNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, prefix: str = "") -> None:
        super().__init__()
        self.encoder = encoder

        # Constants
        Vt = 10.0
        tm = 100.0
        tf = 20.0
        Tsyn = 1.0
        Tmin = encoder.Tmin

        we = Vt
        wi = -Vt
        wacc = (Vt * tm) / encoder.Tcod
        gmult = (Vt * tm) / tf

        # Neurons
        self.input1 = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "input1")
        self.input2 = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "input2")
        self.first1 = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "first1")
        self.last1 = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "last1")
        self.acc_log1 = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "acc_log1")

        self.first2 = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "first2")
        self.last2 = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "last2")
        self.acc_log2 = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "acc_log2")

        self.sync = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "sync")
        self.acc_exp = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "acc_exp")
        self.output = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "output")

        self.connect_neurons(self.input1, self.first1, "V", we, Tsyn)
        self.connect_neurons(self.input1, self.last1, "V", 0.5 * we, Tsyn)
        self.connect_neurons(self.first1, self.first1, "V", wi, Tsyn)
        self.connect_neurons(self.first1, self.acc_log1, "ge", wacc, Tsyn + Tmin)
        self.connect_neurons(self.last1, self.acc_log1, "ge", -wacc, Tsyn)
        self.connect_neurons(self.last1, self.sync, "V", 0.5 * we, Tsyn)

        self.connect_neurons(self.input2, self.first2, "V", we, Tsyn)
        self.connect_neurons(self.input2, self.last2, "V", 0.5 * we, Tsyn)
        self.connect_neurons(self.first2, self.first2, "V", wi, Tsyn)
        self.connect_neurons(self.first2, self.acc_log2, "ge", wacc, Tsyn + Tmin)
        self.connect_neurons(self.last2, self.acc_log2, "ge", -wacc, Tsyn)
        self.connect_neurons(self.last2, self.sync, "V", 0.5 * we, Tsyn)

        self.connect_neurons(self.acc_log1, self.acc_log2, "gf", gmult, Tsyn)
        self.connect_neurons(self.acc_log1, self.acc_log2, "gate", 1, Tsyn)

        self.connect_neurons(self.acc_log2, self.acc_exp, "gate", -1, Tsyn)
        self.connect_neurons(self.acc_log2, self.acc_exp, "ge", wacc, Tsyn)
        self.connect_neurons(self.acc_log2, self.output, "V", we, 2 * Tsyn)

        self.connect_neurons(self.sync, self.acc_log1, "gf", gmult, Tsyn)
        self.connect_neurons(self.sync, self.acc_log1, "gate", 1, Tsyn)
        self.connect_neurons(self.sync, self.acc_exp, "gate", 1, 3 * Tsyn)
        self.connect_neurons(self.sync, self.acc_exp, "gf", gmult, 3 * Tsyn)

        self.connect_neurons(self.acc_exp, self.output, "V", we, Tmin + Tsyn)


if __name__ == "__main__":
    from stick_emulator.simulator import Simulator

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = MultiplierNetwork(encoder)

    val1 = 0.5
    val2 = 0.01
    true_product = val1 * val2

    sim = Simulator(net, encoder, dt=0.01)

    # Apply both input values
    sim.apply_input_value(val1, neuron=net.input1, t0=10)
    sim.apply_input_value(val2, neuron=net.input2, t0=10)

    # Simulate long enough to see output
    sim.simulate(simulation_time=400)

    spikes = sim.spike_log.get(net.output.uid, [])

    if len(spikes) >= 2:
        interval = spikes[1] - spikes[0]
        decoded = encoder.decode_interval(interval)
        print(f"✅ Input: {val1} × {val2}")
        print(f"✅ Expected: {true_product:.4f}")
        print(f"✅ Output spike interval: {interval:.3f} ms")
        print(f"✅ Decoded value: {decoded:.4f}")
        print(f"✅ Error: {abs(decoded - true_product):.4f}")
    else:
        print(f"❌ Output spike missing or incomplete: {spikes}")
