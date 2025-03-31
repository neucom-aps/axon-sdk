from primitives import SpikingNetworkModule, ExplicitNeuron, DataEncoder

class SubtractorSimpleNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder):
        super().__init__()
        self.encoder = encoder

        # Parameters
        Vt = 10.0
        tm = 100.0
        tf = 20.0
        Tsyn = 1.0
        Tneu = 0.01
        Tmin = encoder.Tmin

        we = Vt
        wi = -Vt

        self.input1 = ExplicitNeuron(Vt, tm, tf, neuron_id="input1")
        self.input2 = ExplicitNeuron(Vt, tm, tf, neuron_id="input2")
        self.sync1 = ExplicitNeuron(Vt, tm, tf, neuron_id="sync1")
        self.sync2 = ExplicitNeuron(Vt, tm, tf, neuron_id="sync2")
        self.inb1 = ExplicitNeuron(Vt, tm, tf, neuron_id="inb1")
        self.inb2 = ExplicitNeuron(Vt, tm, tf, neuron_id="inb2")
        self.output_plus = ExplicitNeuron(Vt, tm, tf, neuron_id="output_plus")
        self.output_minus = ExplicitNeuron(Vt, tm, tf, neuron_id="output_minus")

        self.add_neurons([
            self.input1, self.input2,
            self.sync1, self.sync2,
            self.inb1, self.inb2,
            self.output_plus, self.output_minus
        ])

        self.connect_neurons(self.input1, self.sync1, 'V', 0.5 * we, Tsyn)
        self.connect_neurons(self.input2, self.sync2, 'V', 0.5 * we, Tsyn)

        self.connect_neurons(self.sync1, self.inb1, 'V', we, Tsyn)
        self.connect_neurons(self.sync1, self.inb2, 'V', wi, Tsyn)
        self.connect_neurons(self.sync1, self.output_plus, 'V', we, Tmin + (3*Tsyn)+(2*Tneu))
        self.connect_neurons(self.sync1, self.output_minus, 'V', we, (3*Tsyn)+(2*Tneu))

        self.connect_neurons(self.sync2, self.inb1, 'V', wi, Tsyn)
        self.connect_neurons(self.sync2, self.inb2, 'V', we, Tsyn)
        self.connect_neurons(self.sync2, self.output_plus, 'V', we, (3*Tsyn)+(2*Tneu))
        self.connect_neurons(self.sync2, self.output_minus, 'V', we, Tmin + (3*Tsyn)+ (2*Tneu))

        self.connect_neurons(self.inb1, self.output_plus, 'V', 2*wi, Tsyn)
        self.connect_neurons(self.inb2, self.output_minus, 'V', 2*wi, Tsyn)

        self.connect_neurons(self.output_plus, self.inb2, 'V', 0.5 * we, Tsyn)
        self.connect_neurons(self.output_minus, self.inb1, 'V', 0.5 * we, Tsyn)


if __name__ == '__main__':
    from simulator import Simulator
    encoder = DataEncoder()

    sub_net = SubtractorSimpleNetwork(encoder)
    sim = Simulator(sub_net, encoder, dt=.001)

    x, y = 1, 1
    sim.apply_input_value(value=x, neuron=sub_net.input1, t0=0)
    sim.apply_input_value(value=y, neuron=sub_net.input2, t0=0)

    sim.simulate(200)
    print(sim.spike_log.get(sub_net.output_plus.id, []))



