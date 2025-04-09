from stick_emulator.primitives import (
    SpikingNetworkModule,
    DataEncoder,
    ExplicitNeuron,
)
from stick_emulator.networks import MemoryNetwork


class SignedMemoryNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder) -> None:
        super().__init__()

        Vt = 10.0
        tm = 100.0
        tf = 20.0
        we = Vt
        wi = -Vt
        Tsyn = 1.0

        # Main neurons
        input_pos = ExplicitNeuron(Vt, tm, tf, neuron_id="input+")
        input_neg = ExplicitNeuron(Vt, tm, tf, neuron_id="input-")
        ready_pos = ExplicitNeuron(Vt, tm, tf, neuron_id="ready+")
        ready_neg = ExplicitNeuron(Vt, tm, tf, neuron_id="ready-")
        recall = ExplicitNeuron(Vt, tm, tf, neuron_id="recall")
        output_pos = ExplicitNeuron(Vt, tm, tf, neuron_id="output+")
        output_neg = ExplicitNeuron(Vt, tm, tf, neuron_id="output-")
        ready_out = ExplicitNeuron(Vt, tm, tf, neuron_id="ready")

        # Internal memory block
        self.mem = MemoryNetwork(encoder)
        self.add_subnetwork(self.mem)

        # Add all external neurons
        self.add_neurons(
            [
                input_pos,
                input_neg,
                ready_pos,
                ready_neg,
                recall,
                output_pos,
                output_neg,
                ready_out,
            ]
        )

        # === Connections from inputs to ready
        self.connect_neurons(input_pos, ready_pos, "V", we, Tsyn)
        self.connect_neurons(input_neg, ready_neg, "V", we, Tsyn)
        self.connect_neurons(input_pos, ready_neg, "V", 0.25 * we, Tsyn)
        self.connect_neurons(input_neg, ready_pos, "V", 0.25 * we, Tsyn)

        # Inhibit opposite ready neuron
        self.connect_neurons(ready_pos, ready_neg, "V", 0.5 * wi, Tsyn)
        self.connect_neurons(ready_neg, ready_pos, "V", 0.5 * wi, Tsyn)

        # === Routing recall spike to both ready neurons
        self.connect_neurons(recall, ready_pos, "V", 0.5 * we, Tsyn)
        self.connect_neurons(recall, ready_neg, "V", 0.5 * we, Tsyn)

        # === Route input to internal memory
        self.connect_neurons(input_pos, self.mem.input, "V", we, Tsyn)
        self.connect_neurons(input_neg, self.mem.input, "V", we, Tsyn)

        # === Route recall to internal memory recall
        self.connect_neurons(recall, self.mem.recall, "V", we, Tsyn)

        # === Route memory output to both output pos and neg
        self.connect_neurons(self.mem.output, output_pos, "V", we, Tsyn)
        self.connect_neurons(self.mem.output, output_neg, "V", we, Tsyn)

        # === Inhibit the wrong output
        self.connect_neurons(ready_pos, output_neg, "V", 2 * wi, Tsyn)
        self.connect_neurons(ready_neg, output_pos, "V", 2 * wi, Tsyn)

        # === Signal output ready after memory recall
        self.connect_neurons(self.mem.output, ready_out, "V", we, Tsyn)

        # External references
        self.input_pos = input_pos
        self.input_neg = input_neg
        self.recall = recall
        self.output_pos = output_pos
        self.output_neg = output_neg
        self.ready = ready_out


if __name__ == "__main__":
<<<<<<< HEAD
    from stick_emulator import Simulator
=======
    from stick_emulator.simulator import Simulator
>>>>>>> 77c57fee7690484bebf3aa51ef5278843ac2c26b

    encoder = DataEncoder()
    net = SignedMemoryNetwork(encoder)
    sim = Simulator(net, encoder, dt=0.01)

    # ============================
    # Set Input
    # ============================
    input_value = -0.5  # try changing this between -1.0 and 1.0

    abs_val = abs(input_value)
    t0 = 0

    if input_value >= 0:
        sim.apply_input_value(abs_val, net.input_pos, t0=t0)
    else:
        sim.apply_input_value(abs_val, net.input_neg, t0=t0)

    # ============================
    # Trigger recall after some delay
    # ============================
    recall_time = 100
    sim.apply_input_spike(net.recall, t=recall_time)

    # ============================
    # Run simulation
    # ============================
    sim.simulate(300)

    # ============================
    # Decode output
    # ============================
    pos_spikes = sim.spike_log.get(net.output_pos.uid, [])
    neg_spikes = sim.spike_log.get(net.output_neg.uid, [])

    def try_decode(spikes):
        if len(spikes) >= 2:
            interval = spikes[1] - spikes[0]
            value = encoder.decode_interval(interval)
            return interval, value
        return None, None

    interval_pos, val_pos = try_decode(pos_spikes)
    interval_neg, val_neg = try_decode(neg_spikes)

    print("\n===========================")
    print(f"Input value: {input_value:.3f}")

    if val_pos is not None:
        print(f"✅ Output + spike interval: {interval_pos:.3f} ms")
        print(f"✅ Output + recalled value: {val_pos:.3f}")
    elif val_neg is not None:
        print(f"✅ Output - spike interval: {interval_neg:.3f} ms")
        print(f"✅ Output - recalled value: {-val_neg:.3f}")
    else:
        print("❌ No valid output spikes found.")
