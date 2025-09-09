from axon_sdk.primitives import SpikingNetworkModule
from axon_sdk import PredSimulator

from axon_sdk.networks import (
    AdderNetwork,
    SignedMultiplierNormNetwork,
    SignFlipperNetwork,
    DivNetwork,
)
from ..compilation.compiler import InjectorNetwork
from .power_metrics import estimate_performance, estimate_power_and_energy


def benchmark_simulation(sim: PredSimulator) -> None:
    if sim.finished is False:
        raise ValueError("Simulation not executed. Run it before!")
    
    print("\n")
    print("--------------------------------------------------")
    print("---------- SIMULATION BENCHMARK REPORT -----------")
    print("--------------------------------------------------")
    print("\n")

    report_neuron_usage(sim.net, print_depth=0)
    report_spike_usage_for_simulation(sim)
    report_energy_and_latency_for_simulation(sim)


def benchmark_net(net: SpikingNetworkModule) -> None:
    print("IMP: Benchmarks on a net are estimations!")
    print("To benchmark actual runtime behaviour, use 'benchmark_simulation()'")

    report_neuron_usage(net)
    report_spike_estimation_for_net(net)
    report_energy_and_latency_estimation_for_net(net)


def report_neuron_usage(net: SpikingNetworkModule, print_depth=1):
    print("--------------- NEURON USAGE REPORT --------------")
    _print_neurons(net, max_recursion=print_depth)
    print(f"-> Total: {len(net.neurons)} neurons")
    print("\n")


def _print_neurons(net: SpikingNetworkModule, max_recursion=1, indent=0, rec_level=0):
    """Print the neuron count for the module and its submodules."""

    print(" " * indent + f"{net.uid}: {len(net.neurons)} neurons")

    if rec_level < max_recursion:
        if len(net.subnetworks) != 0:
            print(" " * (indent + 4) + f"own {len(net.top_module_neurons)}")

        for subnet in net.subnetworks:
            _print_neurons(subnet, indent=indent + 4, rec_level=rec_level + 1)


def _module_to_total_spikes(mod: SpikingNetworkModule) -> int:
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


def _module_to_spikes(mod: SpikingNetworkModule) -> dict[str, int]:
    # Maps each module to the count of processed spikes within each module
    if isinstance(mod, InjectorNetwork):
        return {"V": 0, "ge": 0, "gf": 0, "gm": 0}
    elif isinstance(mod, SignFlipperNetwork):
        return {"V": 2, "ge": 0, "gf": 0, "gm": 0}
    elif isinstance(mod, SignedMultiplierNormNetwork):
        return {"V": 42, "ge": 9, "gf": 3, "gm": 4}
    elif isinstance(mod, AdderNetwork):
        return {"V": 76, "ge": 16, "gf": 0, "gm": 0}
    elif isinstance(mod, DivNetwork):
        return {"V": 48, "ge": 5, "gf": 3, "gm": 4}
    else:
        raise ValueError(f"Unknown number of spikes for module {mod}")


def report_spike_estimation_for_net(net: SpikingNetworkModule):
    """
    To be used with networks produced by the compiler.
    IMP: Will not be accurate if net has top level neurons
    """
    print("--------- SPIKE COUNT REPORT (ESTIMATION) --------")

    if len(net.top_module_neurons) != 0:
        print("IMP: Spike count estimation might not be accurate")
        print("     Noticed your net has neurons in the top module")
        print("     Spikes in top module neurons are not counted")
        print("     Use estimation with care.")

    spikes_count = 0

    indent = 4
    for subnet in net.subnetworks:
        spikes = _module_to_total_spikes(subnet)
        spikes_count += spikes
        print(" " * indent + f"{subnet.uid}: {spikes} spikes")

    print(f"-> Total: {spikes_count} spikes")
    print("\n")


def report_spike_usage_for_simulation(sim: PredSimulator):
    """
    To be used after a simulation has ben finalized
    """
    if sim.finished is False:
        raise ValueError("Simulation not executed. Run it before!")
    print("--------------- SPIKE COUNT REPORT ---------------")

    spikes_count = sum(len(d) for d in sim.spike_log.values())

    print(f"-> Total: {spikes_count} spikes")
    print("\n")


def report_energy_and_latency_estimation_for_net(net: SpikingNetworkModule) -> None:
    """
    Estimation because it does not use runtime info. about number of spikes. Instead, it
    uses precomputed spike counts for each module.

    IMP: Intended to be used with networks produced by the compiler since
    `_module_to_spikes` map is only available for certain modules.
    """

    v_spikes = 0
    ge_spikes = 0
    gf_spikes = 0
    gm_spikes = 0

    for subnet in net.subnetworks:
        v_spikes += _module_to_spikes(subnet)["V"]
        ge_spikes += _module_to_spikes(subnet)["ge"]
        gf_spikes += _module_to_spikes(subnet)["gf"]
        gm_spikes += _module_to_spikes(subnet)["gm"]

    print("------ ENERGY & LATENCY REPORT (ESTIMATION) ------")
    _report_energy_and_latency(v_spikes, ge_spikes, gf_spikes, gm_spikes)


def report_energy_and_latency_for_simulation(sim: PredSimulator) -> None:

    v_spikes = sim._processed_synapses_log["V"]
    ge_spikes = sim._processed_synapses_log["ge"]
    gf_spikes = sim._processed_synapses_log["gf"]
    gm_spikes = sim._processed_synapses_log["gm"]

    print("------------ ENERGY & LATENCY REPORT -------------")
    _report_energy_and_latency(v_spikes, ge_spikes, gf_spikes, gm_spikes)


def _report_energy_and_latency(
    v_spikes: int, ge_spikes: int, gf_spikes: int, gm_spikes: int
) -> None:

    perf = estimate_performance(
        num_v_spike_updates=v_spikes,
        num_ge_spike_updates=ge_spikes,
        num_gf_spike_updates=gf_spikes,
        num_gm_spike_updates=gm_spikes,
        predictive_search_timesteps=5000,
        clock_speed_mhz=200,
    )

    energy_estimat = estimate_power_and_energy(perf=perf, clock_speed_mhz=200)

    print("-------")
    print(f"Predictive logic report:")
    print(f"V-type updates: {v_spikes}")
    print(f"ge-type updates: {ge_spikes}")
    print(f"gf-type updates: {gf_spikes}")
    print(f"gm-type updates: {gm_spikes}")
    print("-------")
    print(f"Latency per iteration: {perf['time_seconds']} s")
    print(f"Energy per iteration:  {energy_estimat['E_total']}")
    print(f"Power:                 {energy_estimat['P_total']}")
    print("-------")
    print(f"P_leak:                {energy_estimat['P_leak']}")
    print(f"P_idle:                {energy_estimat['P_idle']}")
    print(f"P_dynamic:             {energy_estimat['P_dynamic']}")
    print(f"I_total:               {energy_estimat['I_total']}")
    print(f"r_SOP:                 {energy_estimat['r_SOP']}")
    print("\n")
