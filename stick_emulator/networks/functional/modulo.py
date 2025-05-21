from stick_emulator import Simulator
from stick_emulator.primitives import SpikingNetworkModule, DataEncoder


class ModNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, n: int, module_name=None):

        super().__init__(module_name)
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

        ge_w = (Vt * tm) / (n * self.encoder.Tcod - Tneu)

        # BRANCHING
        # Branching logic on the input to start and stop the pulser based on input interval
        self.input = self.add_neuron(Vt, tm, tf, neuron_name="input")
        self.first = self.add_neuron(Vt, tm, tf, neuron_name="first")
        self.last = self.add_neuron(Vt, tm, tf, neuron_name="last")
        # PULSER
        self.pulser = self.add_neuron(Vt, tm, tf, neuron_name="interval_pulser")
        # GATING
        # Includes ready and gating neuron
        self.gate = self.add_neuron(Vt, tm, tf, neuron_name="gate")
        self.ready = self.add_neuron(Vt, tm, tf, neuron_name="ready")
        # Second gating mechanism for coincidence detection
        self.zgate = self.add_neuron(Vt, tm, tf, neuron_name="zero_gate")
        self.conicidence = self.add_neuron(
            Vt, tm, tf, neuron_name="concidence_detector"
        )
        # OUTPUT
        self.acc = self.add_neuron(Vt, tm, tf, neuron_name="acc")
        self.recall = self.add_neuron(Vt, tm, tf, neuron_name="recall")
        self.output = self.add_neuron(Vt, tm, tf, neuron_name="output")

        # Branching logic synapses
        self.connect_neurons(self.input, self.first, "V", we, Tneu + Tmin)
        self.connect_neurons(self.input, self.last, "V", 0.5 * we, Tneu)
        self.connect_neurons(self.first, self.first, "V", wi, Tneu)
        # The fist neuron start integration, the last stops it
        self.connect_neurons(self.first, self.pulser, "V", we, Tneu)
        self.connect_neurons(self.last, self.pulser, "ge", -ge_w, Tneu)
        # Self-sustaining activity in the pulser (stopped by the "last" neuron in branching)
        self.connect_neurons(self.pulser, self.pulser, "ge", ge_w, Tneu)

        # Gating logic
        # Final spike of input intervals turns gate ON
        self.connect_neurons(self.last, self.ready, "V", we, Tneu)
        self.connect_neurons(self.ready, self.gate, "V", 0.5 * we, Tneu)
        # Once gate is ON, pulser spikes can get through.
        self.connect_neurons(self.pulser, self.gate, "V", 0.5 * we, Tneu)
        self.connect_neurons(self.pulser, self.gate, "V", 0.5 * wi, 3 * Tneu)

        # Zero gating logic
        # Conincident spikes turn zero gate on
        self.connect_neurons(self.last, self.conicidence, "V", 0.5 * we, Tneu)
        self.connect_neurons(self.pulser, self.conicidence, "V", 0.5 * we, Tneu)
        # This prevents periodical pulses from opening the zero gate
        self.connect_neurons(
            self.pulser, self.conicidence, "V", -0.5 * we, Tneu + Tneu
        )
        self.connect_neurons(self.conicidence, self.zgate, "V", 0.5 * we, Tneu)
        # Once gate is ON, pulser spikes can get through.
        self.connect_neurons(self.pulser, self.zgate, "V", 0.5 * we, Tneu)
        self.connect_neurons(self.pulser, self.zgate, "V", 0.5 * wi, 2 * Tneu)
        # Zero gate provide first output interval
        self.connect_neurons(self.zgate, self.output, "V", we, Tneu)

        # Ready neuron restarts integration from last mod value
        self.connect_neurons(self.ready, self.pulser, "ge", ge_w, Tmin)
        # Accumulator inverts the value
        self.connect_neurons(self.ready, self.acc, "ge", ge_w, Tmin + 2 * Tneu)
        # Gate resets integration in pulser and acc
        self.connect_neurons(self.gate, self.pulser, "ge", -ge_w, Tneu)
        self.connect_neurons(self.gate, self.acc, "ge", -ge_w, Tneu)
        # Gate triggers recall
        self.connect_neurons(self.gate, self.recall, "V", we, Tmin)
        # Recall restart accumulator
        self.connect_neurons(self.recall, self.acc, "ge", ge_w, Tneu)
        # Recall provide first output spike
        self.connect_neurons(self.recall, self.output, "V", we, Tneu)
        # Accumulator ends with second output spike
        self.connect_neurons(self.acc, self.output, "V", we, Tmin)


if __name__ == "__main__":
    from stick_emulator import Simulator

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = ModNetwork(encoder, n=0.032, module_name="mod")
    sim = Simulator(net, encoder, dt=0.01)

    sim.apply_input_value(0.048, net.input, t0=0.0)

    sim.simulate(simulation_time=300)

    output_spikes = sim.spike_log.get(net.output.uid, [])
    interval = output_spikes[1] - output_spikes[0]
    decoded = encoder.decode_interval(interval)

    print(decoded)
