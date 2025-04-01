from stick_emulator.primitives import (
    SpikingNetworkModule,
    DataEncoder,
    ExplicitNeuron,
)
from stick_emulator.networks import MemoryNetwork


class SynchronizerNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, N: int):
        super().__init__()
        self.encoder = encoder
        self.N = N

        Vt = 10.0
        tm = 100.0
        tf = 20.0

        we = Vt
        Tsyn = 1.0

        # Create sync neuron
        self.sync = ExplicitNeuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_id="sync"
        )
        self.add_neurons(self.sync)

        self.input_neurons = []
        self.output_neurons = []
        self.memory_blocks = []

        for i in range(N):
            # Input and output interface neurons
            input_neuron = ExplicitNeuron(
                Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_id=f"input_{i}"
            )
            output_neuron = ExplicitNeuron(
                Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_id=f"output_{i}"
            )
            self.add_neurons([input_neuron, output_neuron])
            self.input_neurons.append(input_neuron)
            self.output_neurons.append(output_neuron)

            # Memory network block
            memory = MemoryNetwork(encoder, f"-{i}")
            self.add_subnetwork(memory)
            self.memory_blocks.append(memory)

            # Connect input to memory.input
            self.connect_neurons(input_neuron, memory.input, "V", we, Tsyn)

            # Connect memory.output to output neuron
            self.connect_neurons(memory.output, output_neuron, "V", we, Tsyn)

            # Connect memory.ready to sync
            self.connect_neurons(
                memory.ready, self.sync, "V", (we / N) + 0.1, Tsyn
            )

            # Connect sync to memory.recall
            self.connect_neurons(self.sync, memory.recall, "V", we, Tsyn)


if __name__ == "__main__":

    from stick_emulator.simulator import Simulator
    from stick_emulator.primitives import DataEncoder

    encoder = DataEncoder()
    N = 100  # Number of synchronized inputs
    values = [0.5 for i in range(N)]  # N values between 0–1

    # Initialize network and simulator
    syncnet = SynchronizerNetwork(encoder, N=N)
    sim = Simulator(net=syncnet, encoder=encoder, dt=0.01)

    # Apply inputs using simulator-provided method
    t0 = 10  # Time to begin encoding spikes
    for i, val in enumerate(values):
        sim.apply_input_value(value=val, neuron=syncnet.input_neurons[i], t0=t0)

    # Run simulation
    sim.simulate(simulation_time=200)
    print(sim.spike_log)

    # Collect and decode outputs
    for i, out_neuron in enumerate(syncnet.output_neurons):
        print(out_neuron.id)
        spikes = sim.spike_log.get(out_neuron.id, [])
        if len(spikes) >= 2:
            interval = spikes[1] - spikes[0]
            decoded = encoder.decode_interval(interval)
            print(
                f"✅ Output[{i}] | Interval: {interval:.3f} ms | Decoded: {decoded:.3f}"
            )
        else:
            print(f"❌ Output[{i}] missing second spike: {spikes}")
