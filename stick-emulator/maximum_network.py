from primitives import SpikingNetworkModule, DataEncoder, ExplicitNeuron


class MaximumNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder) -> None:
        super().__init__()
        Vt = 10.0
        tm = 100.0
        tf = 20.0

        we = Vt
        wi = -Vt
        Tsyn = 1.0
        Tneu = 0.01
        Tmin = encoder.Tmin
        wacc = Vt * tm / encoder.Tmax

        # Create Neurons
        input1 = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id="input1")
        input2 = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id="input2")

        larger1 = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id="smaller1")
        larger2 = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id="smaller2")
        output = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id="output")

        # Connections from first input
        self.connect_neurons(input1, larger2, "V", 0.5 * we, Tsyn)
        self.connect_neurons(input1, output, "V", 0.5 * we, Tsyn)

        # Connections from second input
        self.connect_neurons(input2, larger1, "V", 0.5 * we, Tsyn)
        self.connect_neurons(input2, output, "V", 0.5 * we, Tsyn)

        # Connections from second larger
        self.connect_neurons(larger2, larger1, "V", wi, Tsyn)

        # Connections from first larger
        self.connect_neurons(larger1, larger2, "V", wi, Tsyn)

        # Register neurons
        self.add_neurons([input1, input2, larger1, larger2, output])

        # External references
        self.input_1 = input1
        self.input_2 = input2
        self.smaller_1 = larger1
        self.smaller_2 = larger2
        self.output = output


if __name__ == "__main__":
    from simulator import Simulator

    first_value = 0.25
    second_value = 0.75

    encoder = DataEncoder(Tcod=100)
    maxnet = MaximumNetwork(encoder)

    # Set up simulator
    sim = Simulator(net=maxnet, encoder=encoder, dt=0.01)

    # Apply encoded inputs to both elements
    sim.apply_input_value(value=first_value, neuron=maxnet.input_1, t0=0)
    sim.apply_input_value(value=second_value, neuron=maxnet.input_2, t0=0)

    # Run simulation for enough time to capture output
    sim.simulate(simulation_time=500)

    # Retrieve and decode output
    output_spikes = sim.spike_log.get(maxnet.output.uid, [])
    if len(output_spikes) >= 2:
        out_val = encoder.decode_interval(output_spikes[1] - output_spikes[0])
        print(f"✅ Input values: {first_value:.3f}, {second_value:.3f}")
        print(f"✅ Maximum recalled value: {out_val:.3f}")
        # print(f"✅ Interval: {output_spikes[1] - output_spikes[0]:.3f} ms")
    else:
        print(f"❌ Output spike missing or incomplete: {output_spikes}")
