from .elements import Synapse, ExplicitNeuron

from typing import Optional


def flatten_nested_list(nested_list: list) -> list:
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_nested_list(item))
        else:
            flat_list.append(item)
    return flat_list


class SpikingNetworkModule:
    _instace_count = 0

    def __init__(self, module_name: Optional[str] = None) -> None:
        self._neurons: list[ExplicitNeuron] = []
        self._subnetworks: list[SpikingNetworkModule] = []
        self._uid = f"{module_name + '_' if module_name else ''}(m{SpikingNetworkModule._instace_count})"
        SpikingNetworkModule._instace_count += 1

    @property
    def uid(self) -> str:
        return self._uid

    @property
    def neurons(self) -> list[ExplicitNeuron]:
        total_neurons = []
        total_neurons.extend(self._neurons)
        sub_neurons = flatten_nested_list(
            [subnet.neurons for subnet in self._subnetworks]
        )
        total_neurons.extend(sub_neurons)
        return total_neurons

    def add_neuron(
        self,
        Vt: float,
        tm: float,
        tf: float,
        Vreset: float = 0.0,
        neuron_name: Optional[str] = None,
    ) -> ExplicitNeuron:
        extended_neuron_name = self._uid + "_" + f"{neuron_name if neuron_name else ''}"
        new_neuron = ExplicitNeuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=Vreset, neuron_name=extended_neuron_name
        )
        self._neurons.append(new_neuron)
        return new_neuron

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
