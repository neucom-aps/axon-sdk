from typing import Dict
import math

# Delays per stage (in clock cycles)
delays_dict = {
    "encoding": 2,
    "decoding": 2,
    "scheduling": 1,
    "sram_access": 1,
    "computation": 1,
}


def human_readable(value: float, unit: str = "") -> str:
    prefixes = [
        (1e-12, "p"),
        (1e-9, "n"),
        (1e-6, "μ"),
        (1e-3, "m"),
        (1, ""),
        (1e3, "K"),
        (1e6, "M"),
        (1e9, "G"),
    ]
    for factor, prefix in reversed(prefixes):
        if value >= factor or factor == 1e-12:
            scaled = value / factor
            return f"{scaled:.3f} {prefix}{unit}".strip()

    smalles_prefix_value, smallest_prefix = prefixes[0]
    scaled = value / smalles_prefix_value
    return f"{scaled:.3f} {smallest_prefix}{unit}".strip()


def estimate_performance(
    num_v_spike_updates: int,
    num_ge_spike_updates: int,
    num_gf_spike_updates: int,
    num_gm_spike_updates: int,
    predictive_search_timesteps: int = 500,
    clock_speed_mhz: float = 100.0,
    neuron_core_batchsize: int = 1,  # Parallelism (1 = serial execution)
) -> Dict[str, float]:
    """
    Estimate performance of hardware given number of processed updates of each syn type

    Args:
        num_v_spike_updates: Count of processed V spikes.
        num_ge_spike_updates: Count of processed GE spikes.
        num_gf_spike_updates: Count of processed GF spikes.
        num_gm_spike_updates: Count of processed GM spikes.
        predictive_search_timesteps: Timesteps
        neurons_per_cycle: Number of parallel neuron updates supported per cycle.
        clock_speed_ghz: Clock frequency in GHz.
        include_pipeline_latency: Include encoding/decoding/scheduling overhead per inference.
    """

    # === Estimate total updates per neuron ===

    v_updates = num_v_spike_updates

    log_updates_per_syn = math.ceil(math.log2(predictive_search_timesteps))
    # g-synapses trigger binary search to find next spike time
    g_updates = log_updates_per_syn * (num_ge_spike_updates + num_gf_spike_updates + num_gm_spike_updates)

    total_updates = v_updates + g_updates

    # === Compute cycles from updates ===
    # Time-multiplexed processing: updates / (parallel units) * per-update cost
    updates_per_cycle = neuron_core_batchsize
    updates = math.ceil(total_updates / updates_per_cycle)

    clock_cycles_per_update = (
        delays_dict["sram_access"]
        + delays_dict["computation"]
        + delays_dict["scheduling"]
    )

    update_cycles = updates * clock_cycles_per_update

    pipeline_cycles = (
        delays_dict["encoding"] + delays_dict["decoding"] + delays_dict["scheduling"]
    )

    total_cycles = update_cycles + pipeline_cycles
    time_seconds = total_cycles / (clock_speed_mhz * 1e6)

    return {
        "total_updates": total_updates,
        "total_cycles": total_cycles,
        "time_seconds": time_seconds,
    }


def estimate_power_and_energy(
    perf: Dict[str, float],
    clock_speed_mhz: float,
    E_sop_pj: float = 12,
    P_leak_uw: float = 27,
    P_idle_per_mhz_uw: float = 178,
    V_dd: float = 0.9,
) -> Dict[str, str]:
    """
    Estimate power, energy and current draw of STICK hardware.
    """

    total_updates = perf["total_updates"]
    time_seconds = perf["time_seconds"]

    r_sop = total_updates / time_seconds if time_seconds > 0 else 0
    E_sop_j = E_sop_pj * 1e-12
    P_leak = P_leak_uw * 1e-6
    P_idle = P_idle_per_mhz_uw * clock_speed_mhz * 1e-6
    P_dyn = E_sop_j * r_sop
    P_total = P_leak + P_idle + P_dyn
    E_total = P_total * time_seconds
    I_total = P_total / V_dd

    return {
        "P_leak": human_readable(P_leak, "W"),
        "P_idle": human_readable(P_idle, "W"),
        "P_dynamic": human_readable(P_dyn, "W"),
        "P_total": human_readable(P_total, "W"),
        "E_total": human_readable(E_total, "J"),
        "I_total": human_readable(I_total, "A"),
        "r_SOP": human_readable(r_sop, "SOP/s"),
    }
