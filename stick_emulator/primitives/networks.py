from abc import ABC, abstractmethod
from .elements import Synapse, ExplicitNeuron


def flatten_nested_list(nested_list: list) -> list:
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_nested_list(item))
        else:
            flat_list.append(item)
    return flat_list


class SpikingNetworkModule:
    def __init__(self) -> None:
        self._neurons: list[ExplicitNeuron] = []
        self._subnetworks: list[SpikingNetworkModule] = []

    @property
    def neurons(self) -> list[ExplicitNeuron]:
        total_neurons = []
        total_neurons.extend(self._neurons)
        sub_neurons = flatten_nested_list(
            [subnet.neurons for subnet in self._subnetworks]
        )
        total_neurons.extend(sub_neurons)
        return total_neurons

    @neurons.setter
    def neurons(self, new_neurons: list[ExplicitNeuron]) -> None:
        self._neurons = new_neurons

    def add_neurons(
        self, neurons: ExplicitNeuron | list[ExplicitNeuron]
    ) -> None:
        if isinstance(neurons, ExplicitNeuron):
            self._neurons.append(neurons)
        elif isinstance(neurons, list) and all(
            isinstance(n, ExplicitNeuron) for n in neurons
        ):
            self._neurons.extend(neurons)
        else:
            raise TypeError(
                "All elements in the list must be of type ExplicitNeuron"
            )

    def add_subnetwork(self, subnet: "SpikingNetworkModule") -> None:
        self._subnetworks.append(subnet)

    def connect_neurons(
        self,
        pre_neuron: ExplicitNeuron,
        post_neuron: ExplicitNeuron,
        synapse_type: str,
        weight: float,
        delay: float,
    ):
        synapse = Synapse(
            pre_neuron=pre_neuron,
            post_neuron=post_neuron,
            synapse_type=synapse_type,
            weight=weight,
            delay=delay,
        )
        pre_neuron.out_synapses.append(synapse)


class AbstractSpikingNetwork(ABC):
    def __init__(self):
        self.neurons = {}
        self.synapses = []

    def add_neuron(self, neuron_id, neuron):
        self.neurons[neuron_id] = neuron

    def connect_neurons(
        self, pre_neuron_id, post_neuron_id, synapse_type, weight, delay
    ):
        synapse = Synapse(
            pre_neuron=self.neurons[pre_neuron_id],
            post_neuron=self.neurons[post_neuron_id],
            synapse_type=synapse_type,
            weight=weight,
            delay=delay,
        )
        self.synapses.append(synapse)

    def synapses_from(self, neuron_id):
        is_neuron_id = lambda synapse: synapse.pre_neuron.id is neuron_id
        return list(filter(is_neuron_id, self.synapses))

    @abstractmethod
    def simulate(self, simulation_time, dt):
        pass
