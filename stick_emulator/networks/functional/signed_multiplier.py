from stick_emulator.primitives import (
    SpikingNetworkModule,
    DataEncoder,
    ExplicitNeuron,
)
from stick_emulator.networks import MultiplierNetwork


class SignedMultiplierNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, prefix: str = "") -> None:
        super().__init__()
        self.encoder = encoder

        # Parameters
        Vt = 10.0
        tm = 100.0
        tf = 20.0
        Tsyn = 1.0
        Tmin = encoder.Tmin

        we = Vt
        wi = -Vt

        # Create multiplier network
        self.mn = MultiplierNetwork(encoder, prefix=prefix)
        self.add_subnetwork(self.mn)

        self.input1_plus = ExplicitNeuron(
            Vt, tm, tf, neuron_id=prefix + "input1_plus"
        )
        self.input1_minus = ExplicitNeuron(
            Vt, tm, tf, neuron_id=prefix + "input1_minus"
        )
        self.input2_plus = ExplicitNeuron(
            Vt, tm, tf, neuron_id=prefix + "input2_plus"
        )
        self.input2_minus = ExplicitNeuron(
            Vt, tm, tf, neuron_id=prefix + "input2_minus"
        )

        self.output_plus = ExplicitNeuron(
            Vt, tm, tf, neuron_id=prefix + "output_plus"
        )
        self.output_minus = ExplicitNeuron(
            Vt, tm, tf, neuron_id=prefix + "output_minus"
        )

        self.sign1 = ExplicitNeuron(Vt, tm, tf, neuron_id=prefix + "sign1")
        self.sign2 = ExplicitNeuron(Vt, tm, tf, neuron_id=prefix + "sign2")
        self.sign3 = ExplicitNeuron(Vt, tm, tf, neuron_id=prefix + "sign3")
        self.sign4 = ExplicitNeuron(Vt, tm, tf, neuron_id=prefix + "sign4")

        self.add_neurons(
            [
                self.input1_plus,
                self.input1_minus,
                self.input2_plus,
                self.input2_minus,
                self.output_plus,
                self.output_minus,
                self.sign1,
                self.sign2,
                self.sign3,
                self.sign4,
            ]
        )

        self.connect_neurons(self.input1_plus, self.mn.input1, "V", we, Tsyn)
        self.connect_neurons(self.input1_minus, self.mn.input1, "V", wi, Tsyn)

        self.connect_neurons(self.input2_plus, self.mn.input2, "V", we, Tsyn)
        self.connect_neurons(self.input2_minus, self.mn.input2, "V", wi, Tsyn)

        self.connect_neurons(self.input1_plus, self.sign1, "V", 0.25 * we, Tsyn)
        self.connect_neurons(self.input1_plus, self.sign3, "V", 0.25 * we, Tsyn)

        self.connect_neurons(
            self.input1_minus, self.sign2, "V", 0.25 * we, Tsyn
        )
        self.connect_neurons(
            self.input1_minus, self.sign4, "V", 0.25 * we, Tsyn
        )

        self.connect_neurons(self.input2_plus, self.sign1, "V", 0.25 * we, Tsyn)
        self.connect_neurons(self.input2_plus, self.sign2, "V", 0.25 * we, Tsyn)

        self.connect_neurons(
            self.input2_minus, self.sign3, "V", 0.25 * we, Tsyn
        )
        self.connect_neurons(
            self.input2_minus, self.sign4, "V", 0.25 * we, Tsyn
        )

        self.connect_neurons(self.sign1, self.sign2, "V", wi, Tsyn)
        self.connect_neurons(self.sign1, self.sign3, "V", 0.5 * wi, Tsyn)

        self.connect_neurons(self.sign2, self.sign1, "V", 0.5 * wi, Tsyn)
        self.connect_neurons(self.sign2, self.sign4, "V", 0.5 * wi, Tsyn)

        self.connect_neurons(self.sign3, self.sign1, "V", 0.5 * wi, Tsyn)
        self.connect_neurons(self.sign3, self.sign4, "V", 0.5 * wi, Tsyn)

        self.connect_neurons(self.sign4, self.sign2, "V", 0.5 * wi, Tsyn)
        self.connect_neurons(self.sign4, self.sign3, "V", 0.5 * wi, Tsyn)

        self.connect_neurons(self.mn.output, self.output_plus, "V", we, Tsyn)
        self.connect_neurons(self.mn.output, self.output_minus, "V", we, Tsyn)

        self.connect_neurons(self.sign1, self.output_minus, "V", 2 * wi, Tsyn)
        self.connect_neurons(self.sign2, self.output_plus, "V", 2 * wi, Tsyn)
        self.connect_neurons(self.sign3, self.output_plus, "V", 2 * wi, Tsyn)
        self.connect_neurons(self.sign4, self.output_minus, "V", 2 * wi, Tsyn)


if __name__ == "__main__":
    from stick_emulator.simulator import Simulator

    encoder = DataEncoder()
    signed_mn = SignedMultiplierNetwork(encoder)
