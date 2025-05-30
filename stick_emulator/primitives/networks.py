from .elements import Synapse, ExplicitNeuron

from typing import Optional, Self


def flatten_nested_list(nested_list: list) -> list:
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_nested_list(item))
        else:
            flat_list.append(item)
    return flat_list


class SpikingNetworkModule:
    _global_instance_count = 0

    def __init__(self, module_name: Optional[str] = None) -> None:
        self._neurons: list[ExplicitNeuron] = []
        self._subnetworks: list[Self] = []
        self._instance_count = SpikingNetworkModule._global_instance_count
        if module_name:
            self._uid = f"(m{self.instance_count})_{module_name}"
        else:
            self._uid = f"(m{self.instance_count})"

        SpikingNetworkModule._global_instance_count += 1

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

    @property
    def subnetworks(self) -> list[Self]:
        return self._subnetworks

    @property
    def instance_count(self) -> int:
        return self._instance_count

    def recurse_neurons_with_module_uid(self) -> list[dict[ExplicitNeuron, str]]:
        total_neurons_with_module = []
        total_neurons_with_module = [
            {neuron: self.uid} for neuron in self.top_module_neurons
        ]
        sub_neurons = flatten_nested_list(
            [subnet.recurse_neurons_with_module_uid() for subnet in self._subnetworks]
        )
        total_neurons_with_module.extend(sub_neurons)
        return total_neurons_with_module

    @property
    def neurons_with_module_uid(self) -> dict[ExplicitNeuron, str]:
        dicts = self.recurse_neurons_with_module_uid()
        combined = {}
        for d in dicts:
            combined.update(d)
        return combined

    @property
    def top_module_neurons(self) -> list[ExplicitNeuron]:
        """
        Returns neurons belonging to current module, without taking submodules into account
        """
        return self._neurons

    def add_neuron(
        self,
        Vt: float,
        tm: float,
        tf: float,
        Vreset: float = 0.0,
        neuron_name: Optional[str] = None,
    ) -> ExplicitNeuron:
        new_neuron = ExplicitNeuron(
            Vt=Vt,
            tm=tm,
            tf=tf,
            Vreset=Vreset,
            neuron_name=neuron_name,
            parent_mod_id=self.instance_count,
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
