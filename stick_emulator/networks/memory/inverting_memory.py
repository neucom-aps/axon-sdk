from stick_emulator.primitives import (
    SpikingNetworkModule,
    DataEncoder,
    ExplicitNeuron,
)


class InvertingMemoryNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder) -> None:
        super().__init__()

        Vt = 10.0
        tm = 100.0
        tf = 20.0

        we = Vt
        wi = -Vt
        wacc = Vt * tm / encoder.Tmax
        Tsyn = 1.0
        Tmin = encoder.Tmin
        Tneu = 0.01

        input = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id="input")
        first = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id="first")
        last = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id="last")
        acc = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id="acc")
        recall = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id="recall")
        output = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id="output")

        self.connect_neurons(input, first, "V", we, Tsyn)
        self.connect_neurons(input, last, "V", 0.5 * we, Tsyn)
        self.connect_neurons(first, first, "V", wi, Tsyn)
        self.connect_neurons(first, acc, "ge", wacc, Tsyn + Tmin)
        self.connect_neurons(last, acc, "ge", -wacc, Tsyn)
        self.connect_neurons(recall, acc, "ge", wacc, Tsyn)
        self.connect_neurons(acc, output, "V", we, Tsyn)
        self.connect_neurons(recall, output, "V", we, 2 * Tsyn + Tneu)

        self.add_neurons([input, first, last, acc, recall, output])

        self.input = input
        self.output = output
        self.recall = recall


if __name__ == "__main__":
    from stick_emulator import Simulator

    val = 0.6
    encoder = DataEncoder()
    imn = InvertingMemoryNetwork(encoder)
    sim = Simulator(imn, encoder)
    sim.apply_input_value(value=val, neuron=imn.input, t0=0)
    sim.apply_input_spike(imn.recall, t=200)
    sim.simulate(simulation_time=300)

    output_spikes = sim.spike_log[imn.output.uid]
    out_val = encoder.decode_interval(output_spikes[1] - output_spikes[0])
    print(f"Input val: {val}")
    print(f"Inverted val (1-val): {out_val}")
