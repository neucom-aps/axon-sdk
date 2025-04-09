from stick_emulator.primitives import (
    SpikingNetworkModule,
    DataEncoder,
    ExplicitNeuron,
)
import math


class SignedConstantNetwork(SpikingNetworkModule):
    def __init__(
        self, encoder: DataEncoder, value: float, prefix: str = ""
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.value = value

        Vt = 10.0
        tm = 100.0
        tf = 20.0
        we = Vt
        Tsyn = 1.0
        f_x = (math.fabs(value) * self.encoder.Tcod) + encoder.Tmin

        # Create constant neuron
        self.recall = ExplicitNeuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_id=prefix + "recall"
        )
        self.output_plus = ExplicitNeuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_id=prefix + "output_plus"
        )
        self.output_minus = ExplicitNeuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_id=prefix + "output_minus"
        )
        self.add_neurons(self.recall)
        self.add_neurons(self.output_plus)
        self.add_neurons(self.output_minus)

        # Connect constant neuron to itself with a delay

        if value >= 0:
            self.connect_neurons(self.recall, self.output_plus, "V", we, Tsyn)
            self.connect_neurons(
                self.recall, self.output_plus, "V", we, Tsyn + f_x
            )
        else:
            self.connect_neurons(self.recall, self.output_minus, "V", we, Tsyn)
            self.connect_neurons(
                self.recall, self.output_minus, "V", we, Tsyn + f_x
            )


if __name__ == "__main__":
    from stick_emulator import Simulator

    encoder = DataEncoder()
    value = -1.0  # Constant value between 0–1

    constant_network = SignedConstantNetwork(encoder, value)
    sim = Simulator(constant_network, encoder)
    sim.apply_input_spike(constant_network.recall, t=0)
    sim.simulate(simulation_time=120)
    output_spikes = sim.spike_log[constant_network.output_minus.id]
    print(f"Input value: {value}")
    print(f"Output spikes: {output_spikes}")
