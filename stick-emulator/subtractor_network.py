from primitives import SpikingNetworkModule, ExplicitNeuron, DataEncoder

class SubtractorNetwork(SpikingNetworkModule):
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

        # Create all neurons
        self.input1 = ExplicitNeuron(Vt, tm, tf, neuron_id="input1")
        self.input2 = ExplicitNeuron(Vt, tm, tf, neuron_id="input2")

        self.sync1 = ExplicitNeuron(Vt, tm, tf, neuron_id="sync1")
        self.sync2 = ExplicitNeuron(Vt, tm, tf, neuron_id="sync2")

        self.inb1 = ExplicitNeuron(Vt, tm, tf, neuron_id="inb1")
        self.inb2 = ExplicitNeuron(Vt, tm, tf, neuron_id="inb2")

        self.output_plus = ExplicitNeuron(Vt, tm, tf, neuron_id="output_plus")
        self.output_minus = ExplicitNeuron(Vt, tm, tf, neuron_id="output_minus")

        self.zero = ExplicitNeuron(Vt, tm, tf, neuron_id="zero")

        # Add all neurons
        self.add_neurons([
            self.input1, self.input2,
            self.sync1, self.sync2,
            self.inb1, self.inb2,
            self.output_plus, self.output_minus,
            self.zero
        ])

        # --- Main black synapse pathway ---
        # Inputs to syncs
        self.connect_neurons(self.input1, self.sync1, 'V', 0.5 * we, Tsyn)
        self.connect_neurons(self.input2, self.sync2, 'V', 0.5 * we, Tsyn)

        # sync → inb
        self.connect_neurons(self.sync1, self.inb1, 'V', we, Tsyn)
        self.connect_neurons(self.sync2, self.inb2, 'V', we, Tsyn)

        self.connect_neurons(self.sync1, self.inb2, 'V', wi, Tsyn)
        self.connect_neurons(self.sync2, self.inb1, 'V', wi, Tsyn)

        # sync1/2 → output+/- direct
        self.connect_neurons(self.sync1, self.output_plus, 'V', we, Tmin+(3*Tsyn + 2*Tneu))
        self.connect_neurons(self.sync2, self.output_minus, 'V', we, Tmin+(3*Tsyn + 2*Tneu))

        self.connect_neurons(self.sync2, self.output_plus, 'V', we, 3*Tsyn + 2*Tneu)
        self.connect_neurons(self.sync1, self.output_minus, 'V', we, 3*Tsyn + 2*Tneu)

        # inb1 → output-
        self.connect_neurons(self.inb2, self.output_minus, 'V', 2 * wi,  Tsyn)

        # inb2 → output+
        self.connect_neurons(self.inb1, self.output_plus, 'V', 2 * wi, Tsyn)

        # inb1 → output+ (excitatory leak)
        self.connect_neurons(self.output_plus, self.inb2,  'V', 0.5 * we, Tsyn)
        self.connect_neurons(self.output_minus, self.inb1,  'V', 0.5 * we, Tsyn)


        # zero self excitation and inhibition
        self.connect_neurons(self.zero, self.zero, 'V', we, Tneu)

        # zero to sync1, sync2
        self.connect_neurons(self.sync1, self.zero, 'V', 0.5 * wi, 2*Tneu)
        self.connect_neurons(self.sync2, self.zero, 'V', 0.5 * wi, 2*Tneu)

        self.connect_neurons(self.sync1, self.zero, 'V', 0.5 * we, Tneu)
        self.connect_neurons(self.sync2, self.zero, 'V', 0.5 * we, Tneu)

        # zero to output+
        self.connect_neurons(self.zero, self.output_minus, 'V', 2*wi, Tneu)
        self.connect_neurons(self.zero, self.inb2, 'V', wi, Tneu)

        # Save outputs
        self.output = self.output_plus
        self.output_alt = self.output_minus


from primitives import DataEncoder
from simulator import Simulator
from subtractor_network import SubtractorNetwork  # make sure the class is saved in this file

def decode_interval_to_value(interval, encoder: DataEncoder):
    # Assuming subtractor output interval is like memory: Tmin + value * Tcod
    return (interval - encoder.Tmin) / encoder.Tcod

if __name__ == '__main__':
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = SubtractorNetwork(encoder)
    sim = Simulator(net, encoder, dt=0.01)

    # 👇 Input values to subtract: x1 - x2
    x1 = 1
    x2 = 1

    sim.apply_input_value(value=x1, neuron=net.input1, t0=0)
    sim.apply_input_value(value=x2, neuron=net.input2, t0=0)

    sim.simulate(simulation_time=300)

    # --- Output Decoding ---
    spikes_plus = sim.spike_log.get(net.output.id, [])
    spikes_minus = sim.spike_log.get(net.output_alt.id, [])
    #print(spikes_plus)
    #print(spikes_minus)
    if len(spikes_plus) >= 2:
        interval = spikes_plus[1] - spikes_plus[0]
        decoded = decode_interval_to_value(interval, encoder)
        print(f"✅ Output: x1 - x2 = {x1 - x2:.3f}")
        print(f"✅ output+ interval = {interval:.3f} ms → decoded = {decoded:.3f}")
    elif len(spikes_minus) >= 2:
        interval = spikes_minus[1] - spikes_minus[0]
        decoded = decode_interval_to_value(interval, encoder)
        print(f"✅ Output: x1 - x2 = {x1 - x2:.3f}")
        print(f"✅ output- interval = {interval:.3f} ms → decoded = -{decoded:.3f}")
    elif len(spikes_plus) == 1 and len(spikes_minus) == 0:
        print(f"✅ Output: Equal inputs detected (x1 = x2 = {x1})")
        print(f"✅ Single spike on output+ (zero case) → zero difference")
    else:
        print("❌ No valid output spikes detected")

    # Debug: print spike log
    #print("\nSpike log:")
    #for nid, times in sim.spike_log.items():
    #    print(f"{nid}: {times}")