from stick_emulator import Simulator
from stick_emulator.primitives import SpikingNetworkModule, DataEncoder
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


class OscillatorNetwork(SpikingNetworkModule):
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

        ge_w = (Vt * tm) / (n * self.encoder.Tcod - Tsyn)

        # BRANCHING
        # Branching logic on the input to start and stop the pulser based on input interval
        self.input = self.add_neuron(Vt, tm, tf, neuron_name="input")
        self.first = self.add_neuron(Vt, tm, tf, neuron_name="first")
        self.last = self.add_neuron(Vt, tm, tf, neuron_name="last")
        # PULSER
        self.pulser = self.add_neuron(Vt, tm, tf, neuron_name="interval_pulser")

        # Branching logic synapses
        self.connect_neurons(self.input, self.first, "V", we, Tsyn + Tmin)
        self.connect_neurons(self.input, self.last, "V", 0.5 * we, Tsyn)
        self.connect_neurons(self.first, self.first, "V", wi, Tsyn)
        # The fist neuron start integration, the last stops it
        self.connect_neurons(self.first, self.pulser, "V", we, Tsyn)
        self.connect_neurons(self.last, self.pulser, "ge", -ge_w, 2 * Tsyn)
        # Self-sustaining activity in the pulser (stopped by the "last" neuron in branching)
        self.connect_neurons(self.pulser, self.pulser, "ge", ge_w, Tsyn)


if __name__ == "__main__":
    import pprint
    import numpy as np
    import matplotlib.pyplot as plt
    from stick_emulator import Simulator

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = OscillatorNetwork(encoder, n=0.6, module_name="mod")
    sim = Simulator(net, encoder, dt=0.0001)

    sim.apply_input_value(0.8, net.input, t0=0.0)

    sim.simulate(simulation_time=400)

    pprint.pprint(sim.spike_log)

    voltage = sim.voltage_log["(m0,n3)_interval_pulser"]
    v_array = np.asarray(voltage)
    plt.plot(v_array[:, 1], v_array[:, 0])
    plt.show()

    # plot_chronogram(sim.timesteps, sim.voltage_log, sim.spike_log)

    # output_spikes = sim.spike_log.get(net.output.uid, [])
    # interval = output_spikes[1] - output_spikes[0]
    # decoded = encoder.decode_interval(interval)

    # print(decoded)
