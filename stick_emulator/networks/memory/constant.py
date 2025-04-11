from stick_emulator.primitives import (
    SpikingNetworkModule,
    DataEncoder,
    ExplicitNeuron,
)


class ConstantNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, value: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.value = value

        Vt = 10.0
        tm = 100.0
        tf = 20.0
        we = Vt
        Tsyn = 1.0
        f_x = (value * self.encoder.Tcod) + encoder.Tmin

        # Create constant neuron
        self.recall_neuron = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="recall"
        )
        self.output_neuron = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="output"
        )

        # Connect constant neuron to itself with a delay
        self.connect_neurons(self.recall_neuron, self.output_neuron, "V", we, Tsyn)
        self.connect_neurons(
            self.recall_neuron, self.output_neuron, "V", we, Tsyn + f_x
        )


if __name__ == "__main__":
    from stick_emulator import Simulator

    encoder = DataEncoder()
    value = 0  # Constant value between 0–1

    constant_network = ConstantNetwork(encoder, value)
    sim = Simulator(constant_network, encoder)
    sim.apply_input_spike(constant_network.recall_neuron, t=0)
    sim.simulate(simulation_time=100)
    output_spikes = sim.spike_log[constant_network.output_neuron.uid]
    print(f"Input value: {value}")
    print(f"Output spikes: {output_spikes}")
