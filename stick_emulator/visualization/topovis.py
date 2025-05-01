from stick_emulator.primitives import (
    DataEncoder,
    SpikingNetworkModule,
    ExplicitNeuron,
    Synapse,
)

from stick_emulator.visualization.server import start_server


def post_synapse_neurons(neurons: list[ExplicitNeuron]) -> list[ExplicitNeuron]:
    post_neurons = []
    for neuron in neurons:
        for synapse in neuron.out_synapses:
            post_neurons.append(synapse.post_neuron)
    return post_neurons


def pre_synapse_neurons(
    net: SpikingNetworkModule, target_neurons: list[ExplicitNeuron]
) -> list[ExplicitNeuron]:
    """
    Among the neurons in the net, returns the ones connecting with a neuron in the target neurons
    """
    all_neurons = net.neurons
    selected_neurons = []
    for neuron in all_neurons:
        for syn in neuron.out_synapses:
            if syn.post_neuron in target_neurons:
                selected_neurons.append(neuron)

    return selected_neurons


def get_neurons_to_display(net: SpikingNetworkModule):
    neurons = list(
        set(post_synapse_neurons(net.top_module_neurons))
        | set(net.top_module_neurons)
        | set(pre_synapse_neurons(net, net.top_module_neurons))
    )
    return neurons


def get_synapses_to_display(net: SpikingNetworkModule) -> list[Synapse]:
    selected_synapses = []
    top_mod_neurons = net.top_module_neurons
    for neuron in top_mod_neurons:
        for syn in neuron.out_synapses:
            selected_synapses.append(syn)

    neurons_connecting_top_module_neurons = list(
        set(pre_synapse_neurons(net, top_mod_neurons)) - set(top_mod_neurons)
    )
    for neuron in neurons_connecting_top_module_neurons:
        for syn in neuron.out_synapses:
            if syn.post_neuron in top_mod_neurons:
                selected_synapses.append(syn)

    return selected_synapses


def get_groups_to_display(
    net: SpikingNetworkModule,
) -> list[tuple[ExplicitNeuron, str]]:
    """
    Shows groups for each module that contains neurons which are connected to the top module neurons
    """

    grouped_neurons: dict[ExplicitNeuron, str] = net.neurons_with_module_uid
    top_mod_neurons = net.top_module_neurons
    first_mod_neurons_to_display = list(
        (set(post_synapse_neurons(top_mod_neurons)) - set(top_mod_neurons))
        | (set(pre_synapse_neurons(net, top_mod_neurons)) - set(top_mod_neurons))
    )
    selected_groups = [
        (neuron, grouped_neurons[neuron]) for neuron in first_mod_neurons_to_display
    ]
    return selected_groups


def vis_topology(net: SpikingNetworkModule) -> None:
    neurons_to_display = get_neurons_to_display(net)
    synapses_to_display = get_synapses_to_display(net)
    groups_to_display = get_groups_to_display(net)
    nodes = format_nodes(neurons_to_display)
    edges = format_edges(synapses_to_display)
    groups = format_groups(groups_to_display)

    graph_data = {}
    graph_data["nodes"] = nodes
    graph_data["edges"] = edges
    graph_data["groups"] = groups

    start_server(graph_data)


def format_nodes(neurons: list[ExplicitNeuron]) -> list[dict[str, str]]:
    nodes = []
    for neuron in neurons:
        item = {}
        item["id"] = neuron.uid
        item["label"] = neuron.uid
        nodes.append(item)

    return nodes


def color_for_synapse(synapse_type: str) -> str:
    if synapse_type == "V":
        return "#000000"
    if synapse_type == "ge":
        return "#FF0830"
    if synapse_type == "gf":
        return "#006400"
    if synapse_type == "gate":
        return "#0E1AFE"


def format_edges(synapses: list[Synapse]) -> list[dict[str, str]]:
    edges = []
    for syn in synapses:
        item = {}
        item["source"] = syn.pre_neuron.uid
        item["target"] = syn.post_neuron.uid
        item["label"] = f"({syn.weight:.3f}; {syn.delay:.3f})"
        item["color"] = color_for_synapse(syn.type)
        item["uid"] = syn.uid
        edges.append(item)
    return edges


def format_groups(groups: list[tuple[ExplicitNeuron, str]]) -> list[dict[str, str]]:
    """
    The input groups contain a list of tuples, where each component contains a neuron
    and the uid of their immediately superior module
    """

    def add_element(d, key, value):
        if key not in d:
            d[key] = []
        d[key].append(value)

    groupped_nodes: dict[str, list[ExplicitNeuron]] = {}
    for group in groups:
        add_element(groupped_nodes, group[1], group[0])

    formatted_groups = []
    for key in groupped_nodes:
        formatted_group = {}
        formatted_group["id"] = key
        formatted_group["label"] = key
        formatted_group["nodes"] = [neuron.uid for neuron in groupped_nodes[key]]
        formatted_groups.append(formatted_group)

    return formatted_groups
