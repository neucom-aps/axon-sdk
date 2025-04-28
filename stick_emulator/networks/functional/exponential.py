from stick_emulator.primitives import (
    SpikingNetworkModule,
    DataEncoder,
    ExplicitNeuron,
)
import math

T_F = 50.0


class ExponentialNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, prefix: str = "") -> None:
        super().__init__()
        self.encoder = encoder

        # Parameters
        Vt = 10.0
        tm = 100.0
        tf = T_F
        Tsyn = 1.0
        Tmin = encoder.Tmin

        we = Vt
        wi = -Vt
        gmult = (Vt * tm) / tf
        wacc = Vt * tm / encoder.Tcod  # To ensure V = Vt at Tcod

        # Neurons
        self.input = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "_input")
        self.first = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "_first")
        self.last = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "_last")
        self.acc = self.add_neuron(Vt, tm, tf, neuron_name=prefix + "_acc")
        self.output = self.add_neuron(
            Vt, tm, tf, neuron_name=prefix + "_output"
        )

        # Connections from input neuron
        self.connect_neurons(self.input, self.first, "V", we, Tsyn)
        self.connect_neurons(self.input, self.last, "V", 0.5 * we, Tsyn)

        # Inhibit first after spike
        self.connect_neurons(self.first, self.first, "V", wi, Tsyn)

        # Exponential computation:
        # 1. First spike → apply gf with delay = Tsyn + Tmin
        self.connect_neurons(self.first, self.acc, "gf", gmult, Tsyn + Tmin)
        self.connect_neurons(self.first, self.acc, "gate", 1, Tsyn + Tmin)
        # 2. Last spike → open gate
        self.connect_neurons(self.last, self.acc, "gate", -1.0, Tsyn)

        # 3. Last spike → add ge to trigger spike after ts
        self.connect_neurons(self.last, self.acc, "ge", wacc, Tsyn)

        # Readout to output
        self.connect_neurons(self.acc, self.output, "V", we, Tsyn + Tmin)
        self.connect_neurons(self.last, self.output, "V", we, 2 * Tsyn)

    def get_output_spikes(self):
        return self.output.spike_times


def expected_exp_output_delay(x, Tmin=10.0, Tcod=100.0, tf=T_F):
    Tcod_val = x * Tcod
    try:
        delay = Tcod * math.exp(-Tcod_val / tf)
        Tout = Tmin + delay
        return Tout
    except:
        return float("nan")


def decode_exponential(output_interval, Tmin=10.0, Tcod=100.0, tf=T_F):
    return ((output_interval - Tmin) / Tcod) ** (-tf / Tcod)


if __name__ == "__main__":
    from stick_emulator import Simulator

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = ExponentialNetwork(encoder)

    value = 0.5
    sim = Simulator(net, encoder, dt=0.01)
    sim.apply_input_value(value, neuron=net.input, t0=10)
    sim.simulate(300)

    output_spikes = sim.spike_log.get(net.output.uid, [])
    acc_spikes = sim.spike_log.get(net.acc.uid, [])
    last_spike = sim.spike_log.get(net.last.uid, [None])[-1]
    print(output_spikes[1] - output_spikes[0])
    print(f"Input value: {value}")
    print(f"Expected exp delay: {expected_exp_output_delay(value):.3f} ms")
