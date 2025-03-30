from primitives import ExplicitNeuron, EventQueue, AbstractSpikingNetwork, DataEncoder
from collections import defaultdict

class InvertingMemoryNetwork(AbstractSpikingNetwork):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.event_queue = EventQueue()

        Vt = 10.0
        tm = 100.0
        tf = 20.0

        we = Vt
        wi = -Vt
        wacc = Vt * tm / encoder.Tmax
        Tsyn = 1.0
        Tmin = encoder.Tmin
        Tneu = 0.01

        
        neurons = ['input', 'first', 'last', 'acc', 'recall', 'output']
        for neuron_id in neurons:
            self.add_neuron(neuron_id, ExplicitNeuron(neuron_id=neuron_id, Vt=Vt, tm=tm, tf=tf))

        self.connect_neurons('input', 'first', 'V', we, Tsyn)
        self.connect_neurons('input', 'last', 'V', 0.5 * we, Tsyn)
        self.connect_neurons('first', 'first', 'V', wi, Tsyn)
        self.connect_neurons('first', 'acc', 'ge', wacc, Tsyn + Tmin)
        self.connect_neurons('last', 'acc', 'ge', -wacc, Tsyn)
        self.connect_neurons('recall', 'acc', 'ge', wacc, Tsyn)
        self.connect_neurons('acc', 'output', 'V', we, Tsyn)
        self.connect_neurons('recall', 'output', 'V', we, 2*Tsyn + Tneu)
        
    def simulate(self, simulation_time, dt=0.01):
        spike_log = defaultdict(list)
        timesteps = [i * dt for i in range(int(simulation_time / dt))]
        for t in timesteps:
            events = self.event_queue.pop_events(t)
            for _, neuron_id, syn_type, weight in events:
                self.neurons[neuron_id].receive_synaptic_event(syn_type, weight)

            for neuron_id, neuron in self.neurons.items():
                if neuron.update(dt):
                    spike_log[neuron_id].append(t)
                    neuron.reset()
                    for synapse in self.synapses_from(neuron_id):
                        self.event_queue.add_event(t + synapse.delay, synapse.post_neuron.id, synapse.type, synapse.weight)

        return spike_log

    def apply_input_spikes(self, value):
        spike_interval = self.encoder.encode_value(value)
        for t in spike_interval:
            for synapse in self.synapses_from('input'):
                self.event_queue.add_event(t + synapse.delay, synapse.post_neuron.id, synapse.type, synapse.weight)

    def recall_value(self):
        recall_spike_times = [0]
        for t in recall_spike_times:
            for synapse in self.synapses_from('recall'):
                self.event_queue.add_event(t + synapse.delay, synapse.post_neuron.id, synapse.type, synapse.weight)


if __name__ == '__main__':
    encoder = DataEncoder()
    network = InvertingMemoryNetwork(encoder)

    value = 0.04
    network.apply_input_spikes(value)
    print(f"Input spikes times: {encoder.encode_value(value)[0], encoder.encode_value(value)[1]}")
    print(f"Input spikes interval: {encoder.encode_value(value)[1] - encoder.encode_value(value)[0]}")

    sim_spikes = network.simulate(200, 0.01)

    # Trigger recall
    network.recall_value()
    spikes_after_recall = network.simulate(200, 0.01)

    output_spikes = spikes_after_recall['output']
    decoded_value = encoder.decode_interval(output_spikes[1] - output_spikes[0])

    print("Output spikes times (inverted):", spikes_after_recall['output'])
    print(f"Output spikes interval (inverted): {spikes_after_recall['output'][1] - spikes_after_recall['output'][0]}")

    print(f"Input value: {value}")
    print(f"Decoded value (1-input): {decoded_value}")