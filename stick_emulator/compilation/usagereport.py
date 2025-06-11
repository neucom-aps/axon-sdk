from stick_emulator.primitives import SpikingNetworkModule
from stick_emulator.networks import (
    AdderNetwork,
    SignedMultiplierNormNetwork,
    SignFlipperNetwork,
    DivNetwork,
)
from .compiler import InjectorNetwork
from .power_metrics import estimate_performance, estimate_power_and_energy


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


def module_to_synapses(mod: SpikingNetworkModule) -> dict[str, int]:
    # Note: Added 2 V-synapses per input to account for the connections between modules
    if isinstance(mod, InjectorNetwork):
        return {"V": 2, "ge": 0, "gf": 0, "gate": 0}
    elif isinstance(mod, SignFlipperNetwork):
        return {"V": 4, "ge": 0, "gf": 0, "gate": 0}
    elif isinstance(mod, SignedMultiplierNormNetwork):
        return {"V": 46, "ge": 9, "gf": 3, "gate": 4}
    elif isinstance(mod, AdderNetwork):
        return {"V": 80, "ge": 16, "gf": 0, "gate": 0}
    elif isinstance(mod, DivNetwork):
        return {"V": 52, "ge": 5, "gf": 3, "gate": 4}
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


def report_energy_and_latency_estimation(net: SpikingNetworkModule):
    v_synapses = 0
    ge_synapses = 0
    gf_synapses = 0
    gate_synapses = 0

    for subnet in net.subnetworks:
        v_synapses += module_to_synapses(subnet)["V"]
        ge_synapses += module_to_synapses(subnet)["ge"]
        gf_synapses += module_to_synapses(subnet)["gf"]
        gate_synapses += module_to_synapses(subnet)["gate"]

    perf = estimate_performance(
        num_v_spikes=v_synapses,
        num_ge_spikes=ge_synapses,
        num_gf_spikes=gf_synapses,
        num_gm_spikes=gate_synapses,
        timesteps=5000,
        clock_speed_mhz=200,
    )

    energy_estimat = estimate_power_and_energy(perf=perf, fclk_mhz=200)

    print("---------- ENERGY & LATENCY REPORT ----------")
    print(f"Latency per iteration: {perf['time_seconds']} s")
    print(f"Energy per iteration:  {energy_estimat['E_total']}")
    print(f"Power:                 {energy_estimat['P_total']}")
    print("-------")
    print(f"P_leak:                {energy_estimat['P_leak']}")
    print(f"P_idle:                {energy_estimat['P_idle']}")
    print(f"P_dynamic:             {energy_estimat['P_dynamic']}")
    print(f"I_total:               {energy_estimat['I_total']}")
    print(f"r_SOP:                 {energy_estimat['r_SOP']}")
