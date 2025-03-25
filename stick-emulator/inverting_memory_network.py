from primitives import ExplicitNeuron, EventQueue, AbstractSpikingNetwork, DataEncoder
from collections import defaultdict

class InvertingMemoryNetwork(AbstractSpikingNetwork):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        neurons = ['input', 'first', 'last', 'acc', 'recall', 'output']
        for neuron_id in neurons:
            self.add_neuron(neuron_id, ExplicitNeuron(neuron_id))

        we = 15.0    # Adjusted weights to ensure spikes occur
        wi = 15.0
        wacc = 12.5
        Tsyn = 1 #1.0
        Tmin = encoder.Tmin
        Tneu = 1 #1.0

        self.connect_neurons('input', 'first', 'V', (we, Tsyn))
        self.connect_neurons('input', 'last', 'V', (0.5 * we, Tsyn))
        self.connect_neurons('first', 'first', 'V', (wi, Tsyn))
        self.connect_neurons('first', 'acc', 'ge', (wacc, Tsyn + Tmin))
        self.connect_neurons('last', 'acc', 'ge', (-wacc, Tsyn))
        self.connect_neurons('recall', 'acc', 'ge', (wacc, Tsyn))
        self.connect_neurons('acc', 'output', 'V', (we, Tsyn))
        self.connect_neurons('recall', 'output', 'V', (we, 2*Tsyn + Tneu))

        self.event_queue = EventQueue()

    def simulate(self, simulation_time, dt=1.0):
        for neuron in self.neurons.values():
            neuron.reset()

        spike_log = defaultdict(list)
        for t in range(int(simulation_time)):
            events = self.event_queue.pop_events(t)
            for _, neuron_id, syn_type, weight in events:
                self.neurons[neuron_id].receive_synaptic_event(syn_type, weight)

            for neuron_id, neuron in self.neurons.items():
                if neuron.update(dt):
                    spike_log[neuron_id].append(t)
                    neuron.reset()
                    for post, (syn_type, (weight, delay)) in self.connections[neuron_id].items():
                        self.event_queue.add_event(t + delay, post, syn_type, weight)

        return spike_log

    def apply_input_spikes(self, value):
        spike_interval = self.encoder.encode_value(value)
        for t in spike_interval:
            for post, (syn_type, (weight, delay)) in self.connections['input'].items():
                self.event_queue.add_event(t + delay, post, syn_type, weight)

    def recall_value(self):
        recall_spike_times = [0]
        for t in recall_spike_times:
            for post, (syn_type, (weight, delay)) in self.connections['recall'].items():
                self.event_queue.add_event(t + delay, post, syn_type, weight)


if __name__ == '__main__':
    encoder = DataEncoder()
    network = InvertingMemoryNetwork(encoder)

    # Encode value (e.g., 0.3)
    network.apply_input_spikes(0.3)
    network.simulate(200, 1.0)

    # Trigger recall
    network.recall_value()
    spikes_after_recall = network.simulate(200, 1.0)

    print("Output spike times (inverted):", spikes_after_recall['output'])