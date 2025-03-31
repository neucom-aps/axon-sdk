from primitives import SpikingNetworkModule, DataEncoder, ExplicitNeuron


class LogNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, prefix: str = ""):
        super().__init__()
        self.encoder = encoder

        # Constants
        Vt = 10.0
        tm = 100.0
        tf = 20.0   ## Increasing this makes Log more accurate
        Tsyn = 1.0
        Tmin = encoder.Tmin

        we = Vt
        wi = -Vt
        gmult = (Vt * tm) / tf
        wacc = (Vt * tm) / encoder.Tcod

        # Neurons
        self.input = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id=prefix + "_input")
        self.first = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id=prefix + "_first")
        self.last = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id=prefix + "_last")
        self.acc = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id=prefix + "_acc")
        self.output = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id=prefix + "_output")

        self.add_neurons([self.input, self.first, self.last, self.acc, self.output])

        # Input triggers
        self.connect_neurons(self.input, self.first, 'V', we, Tsyn)
        self.connect_neurons(self.input, self.last, 'V', 0.5 * we, Tsyn)

        # Self-inhibit first after spike
        self.connect_neurons(self.first, self.first, 'V', wi, Tsyn)

        # Encoding into acc from first and last
        self.connect_neurons(self.first, self.acc, 'ge', wacc, Tsyn + Tmin)
        self.connect_neurons(self.last, self.acc, 'gate', 1.0, Tsyn)
        self.connect_neurons(self.last, self.acc, 'gf', gmult, Tsyn)

        # Output driven by acc spike
        self.connect_neurons(self.acc, self.output, 'V', we, Tsyn + Tmin)
        self.connect_neurons(self.last, self.output, 'V', we, 2 * Tsyn)


    def get_output_spikes(self):
        return self.output.spike_times
    
import math

def expected_log_output_delay(x, Tmin=10.0, Tcod=100.0, tf=20.0):
    Tin = Tmin + x * Tcod
    try:
        delay = tf * math.log(Tcod / (Tin - Tmin))
        Tout = Tmin + delay
        return Tout
    except ValueError:
        return float('nan')


if __name__ == '__main__':
    from simulator import Simulator
    from primitives import DataEncoder

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    lognet = LogNetwork(encoder)

    val = 1
    sim = Simulator(lognet, encoder, dt=0.01)
    sim.apply_input_value(val, neuron=lognet.input, t0=10)
    sim.simulate(300)

    output_spikes = sim.spike_log.get(lognet.output.id, [])
    acc_spikes = sim.spike_log.get(lognet.acc.id, [])
    last_spike_time = sim.spike_log.get(lognet.last.id, [None])[-1]

    print(f"Input value: {val}")
    print(f"Expected delay (log({val})): {expected_log_output_delay(val):.3f} ms")

    if len(output_spikes) >= 2:
        interval = output_spikes[1] - output_spikes[0]
        print(f"✅ Output spike interval: {interval:.3f} ms")
    elif len(output_spikes) == 1:
        delay = output_spikes[0] - last_spike_time if last_spike_time else float('nan')
        print(f"✅ Output spike at: {output_spikes[0]:.3f} ms (delay from 'last': {delay:.3f} ms)")
    else:
        print("❌ No output spike detected.")
