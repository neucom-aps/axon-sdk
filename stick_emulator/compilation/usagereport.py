from stick_emulator.primitives import SpikingNetworkModule
from stick_emulator.networks import (
    AdderNetwork,
    SignedMultiplierNormNetwork,
    SignFlipperNetwork,
    DivNetwork
)
from .compiler import InjectorNetwork


def report_neuron_usage(net: SpikingNetworkModule, print_depth=1):
    print("---------- NEURON USAGE REPORT ----------")
    _print_items(net, max_recursion=print_depth)
    print(f"-> Total: {len(net.neurons)} neurons")
    print("\n")


def _print_items(net: SpikingNetworkModule, max_recursion=1, indent=0, rec_level=0):
    """Print the item count for the module and its submodules."""

    print(" " * indent + f"{net.uid}: {len(net.neurons)} neurons")

    if rec_level < max_recursion:
        if len(net.subnetworks) != 0:
            print(" " * (indent + 4) + f"own {len(net.top_module_neurons)}")

        for subnet in net.subnetworks:
            _print_items(subnet, indent=indent + 4, rec_level=rec_level + 1)


def module_to_spikes(mod: SpikingNetworkModule) -> int:
    if isinstance(mod, InjectorNetwork):
        return 2
    elif isinstance(mod, SignFlipperNetwork):
        return 4
    elif isinstance(mod, SignedMultiplierNormNetwork):
        return 29
    elif isinstance(mod, AdderNetwork):
        return 58
    elif isinstance(mod, DivNetwork):
        return 31
    else:
        raise ValueError(f"Unknown number of spikes for module {mod}")


def report_spike_estimation(net: SpikingNetworkModule):
    """
    To be used with networks produced by the compiler.
    IMP: Will not be accurate if net has top level neurons
    """
    print("---------- SPIKE COUNT REPORT ----------")

    if len(net.top_module_neurons) != 0:
        print("IMP: Spike count estimation might not be accurate")
        print("     Noticed your net has neurons in the top module")
        print("     Spikes in top module neurons are not counted")
        print("     Use estimation with care.")

    spikes_count = 0

    indent = 4
    for subnet in net.subnetworks:
        spikes = module_to_spikes(subnet)
        spikes_count += spikes
        print(" " * indent + f"{subnet.uid}: {spikes} spikes")

    print(f"-> Total: {spikes_count} spikes")
    print("\n")
