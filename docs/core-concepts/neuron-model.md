# Neuron Model in Axon

This document details the spiking neuron model used in Axon, which implements the STICK (Spike Time Interval Computational Kernel) computational paradigm. It emphasizes temporal coding, precise spike timing, and synaptic diversity for symbolic computation.

---

##  1. Overview

Axon simulates **event-driven, integrate-and-fire neurons** with:
- **Millisecond-precision spike timing**
- **Multiple synapse types** with distinct temporal effects
- **Explicit gating** to modulate temporal dynamics

The base classes are:
- `AbstractNeuron`: defines core membrane equations
- `ExplicitNeuron`: tracks spike times and enables connectivity
- `Synapse`: defines delayed, typed connections between neurons

---

## 2. Neuron Dynamics

Each neuron maintains four internal state variables:

| Variable | Description |
|----------|-------------|
| `V`      | Membrane potential (mV) |
| `ge`     | Persistent excitatory input (constant) |
| `gf`     | Fast exponential input (gated) |
| `gate`   | Binary gate controlling `gf` integration |

The membrane potential evolves according to:

```math
\tau_m \frac{dV}{dt} = g_e + \text{gate} \cdot g_f
``` 
where:
- `τm` is the membrane time constant
- `g_e` is the persistent excitatory input
- `g_f` is the fast decaying input, gated by `gate`
The neuron spikes when `V` exceeds a threshold `Vt`, at which point it emits a spike and resets its state.
After a spike, the neuron resets:


V → Vreset
ge → 0
gf → 0
gate → 0

This reset guarantees clean integration for subsequent intervals.

---

##  4. Synapse Types

Axon supports four biologically inspired synapse types:

| Type   | Effect                                  |
|--------|------------------------------------------|
| `V`    | Immediate change in membrane: `V += w`   |
| `ge`   | Adds persistent drive: `ge += w`         |
| `gf`   | Adds fast decaying drive: `gf += w`      |
| `gate` | Toggles gate flag (w = ±1) to activate `gf` |

Each synapse also includes a configurable delay, enabling precise temporal computation.

---

##  5. Implementation Summary

### Class: `AbstractNeuron`
- Implements update logic for `ge`, `gf`, and `gate`
- Defines `update_and_spike(dt)` for simulation cycles
- Supports `receive_synaptic_event(type, weight)`

### Class: `ExplicitNeuron`
- Inherits from `AbstractNeuron`
- Tracks:
  - `spike_times[]`
  - `out_synapses[]`
- Implements `reset()` after spike emission

### Class: `Synapse`
- Defines:
  - `pre_neuron`, `post_neuron`
  - `weight`, `delay`, `type`
- Used to construct event-driven spike queues with delay accuracy

---

## 6. Temporal Coding & Integration

This neuron model is designed for **interval-coded** values. Time intervals between spikes directly encode numeric values.

Integration periods in neurons align with computation windows:

- `ge`: accumulates static value during inter-spike interval
- `gf` + `gate`: used for exponential/logarithmic timing
- `V`: compares integrated potential to threshold for spike emission

These dynamics enable symbolic operations such as memory, arithmetic, and differential equation solving.

---

##  7. Numerical Parameters

Typical parameter values used in Axon:

| Parameter | Value    | Meaning                        |
|-----------|----------|--------------------------------|
| `Vt`      | 10.0 mV  | Spiking threshold              |
| `Vreset`  | 0.0 mV   | Voltage after reset            |
| `τm`      | 100.0 ms | Membrane integration constant  |
| `τf`      | 20.0 ms  | Fast synaptic decay constant   |

Units are in milliseconds or millivolts, matching real-time symbolic processing and neuromorphic feasibility.

---

##  8. Benefits of This Model

- **Compact**: Minimal neurons required for functional blocks
- **Precise**: Accurate sub-millisecond spike-based encoding
- **Composable**: Modular design supports hierarchical circuits
- **Hardware-Compatible**: Ported to digital integrate-and-fire cores like ADA

---

##  References

- **Lagorce & Benosman (2015)**: *Spike Time Interval Computational Kernel*  
- **Axon SDK Source**:  
  - Neuron model: `axon/elements.py`  
  - Event logic: `axon/events.py`  
  - Simulator integration: `axon/simulator.py`
