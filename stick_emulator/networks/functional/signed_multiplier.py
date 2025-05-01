from stick_emulator.primitives import (
    SpikingNetworkModule,
    DataEncoder,
)
from stick_emulator.networks import MultiplierNetwork
from typing import Optional


class SignedMultiplierNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, module_name: Optional[str] = None) -> None:
        super().__init__(module_name)
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
        self.mn = MultiplierNetwork(encoder, module_name='mul_net')
        self.add_subnetwork(self.mn)

        self.input1_plus = self.add_neuron(
            Vt, tm, tf, neuron_name="input1_plus"
        )
        self.input1_minus = self.add_neuron(
            Vt, tm, tf, neuron_name="input1_minus"
        )
        self.input2_plus = self.add_neuron(
            Vt, tm, tf, neuron_name="input2_plus"
        )
        self.input2_minus = self.add_neuron(
            Vt, tm, tf, neuron_name="input2_minus"
        )

        self.output_plus = self.add_neuron(
            Vt, tm, tf, neuron_name="output_plus"
        )
        self.output_minus = self.add_neuron(
            Vt, tm, tf, neuron_name="output_minus"
        )

        self.sign1 = self.add_neuron(Vt, tm, tf, neuron_name="sign1")
        self.sign2 = self.add_neuron(Vt, tm, tf, neuron_name="sign2")
        self.sign3 = self.add_neuron(Vt, tm, tf, neuron_name="sign3")
        self.sign4 = self.add_neuron(Vt, tm, tf, neuron_name="sign4")

        self.connect_neurons(self.input1_plus, self.mn.input1, "V", we, Tsyn)
        self.connect_neurons(self.input1_minus, self.mn.input1, "V", wi, Tsyn)

        self.connect_neurons(self.input2_plus, self.mn.input2, "V", we, Tsyn)
        self.connect_neurons(self.input2_minus, self.mn.input2, "V", wi, Tsyn)

        self.connect_neurons(self.input1_plus, self.sign1, "V", 0.25 * we, Tsyn)
        self.connect_neurons(self.input1_plus, self.sign3, "V", 0.25 * we, Tsyn)

        self.connect_neurons(self.input1_minus, self.sign2, "V", 0.25 * we, Tsyn)
        self.connect_neurons(self.input1_minus, self.sign4, "V", 0.25 * we, Tsyn)

        self.connect_neurons(self.input2_plus, self.sign1, "V", 0.25 * we, Tsyn)
        self.connect_neurons(self.input2_plus, self.sign2, "V", 0.25 * we, Tsyn)

        self.connect_neurons(self.input2_minus, self.sign3, "V", 0.25 * we, Tsyn)
        self.connect_neurons(self.input2_minus, self.sign4, "V", 0.25 * we, Tsyn)

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
