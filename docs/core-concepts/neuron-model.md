# Neuron Model

This document details the spiking **neuron model** used in Axon, which implements the [STICK](https://arxiv.org/abs/1507.06222) computation framework. STICK uses temporal coding, precise spike timing, and synaptic diversity for symbolic and deterministic computation.

> [STICK: Spike Time Interval Computational Kernel, A Framework for General Purpose Computation using Neurons, Precise Timing, Delays, and Synchrony](https://arxiv.org/abs/1507.06222).

---

##  Overview

Axon simulates event-driven, **integrate-and-fire neurons** with:
- **Millisecond-precision** spike timing
- Multiple **synapse types** with distinct temporal effects
- Explicit **gating** to modulate temporal dynamics

The base classes are:
- `AbstractNeuron`: defines core membrane equations
- `ExplicitNeuron`: tracks spike times and enables connectivity
- `Synapse`: defines delayed, typed connections between neurons


## Neuron Dynamics

Axon simulates event-driven **non-leaky integrate-and-fire neurons**.

The membrane potentail (`V`) evolves following the differential equations:

\\[ \tau_m \frac{dV}{dt} = g_e + \text{gate} \cdot g_f \\]
\\[ \frac{dg_e}{dt} = 0 \\]
\\[ \tau_f \frac{dg_f}{dt} = -g_f \\]

Each neuron has 4 internal state variables:

| Variable | Description |
|----------|-------------|
| `V`      | Membrane potential |
| `ge`     | Constant input |
| `gf`     | Exponential input |
| `gate`   | Binary gate controlling `gf` |

Each neuron also has 4 internal parameters:

| Parameter | Description |
|----------|-------------|
| `Vt`      | Membrane potential threshold |
| `Vreset`      | Membrane potential set after a reset |
| `tm`     | Timescale of the evolution of the membrane potential|
| `tf`     | Timescale of the evolution of `gf` |

When the membrane potential surpasess a threshold, `V > Vt`, the neuron emits a spike and **resets**:

```text
if V > Vt:
    spike → 1
    V → Vreset
    ge → 0
    gf → 0
    gate → 0
```

##  Synapse Types

The neuron model has 4 synapse types. Each of them affects one of the 4 internal state variables of the neuron receiving the synapse. Synapses have a certain **weight (`w`)**:

| Synapse Type   | Effect                                  | Explanation |
|--------|------------------------------------------|---|
| `V`    | `V += w` | Immediate change in membrane potential |
| `ge`   | `ge += w` | Adds to the constant input |
| `gf`   | `gf += w` | Adds to the exponential input |
| `gate` | `gate += w` | Toggles gate flag (`w = ±1`) to activate `gf` |

Besides it's weight `w`, wach synapse also includes a delay, controlling the the time taken by a spike travelling through the synapse to arrive to the following neuron and affecting it's internal state:

```text
synapse
  | weight
  | delay
```


##  Numerical Parameters

By default, Axon uses the following **numeric values** for the neuron parameters

| Parameter | Numeric value (mV, ms)    | Meaning                        |
|-----------|----------|--------------------------------|
| `Vt`      | 10.0  | Spiking threshold              |
| `Vreset`  | 0.0  | Voltage after reset            |
| `tm`      | 100.0 | Membrane integration constant  |
| `tf`      | 20.0  | Fast synaptic decay constant   |

Units are **milliseconds** for time values and **millivolts** for membrane potential values.

## Neuron Model Animation

This animation demonstrates the evolution over time of an individual neuron when hit by the different input synapses (`V`, `ge`, `gf`, `gate`):

![Neuron](../figs/neural-dynamics.gif)

Synapse-type `ge` produces a linear increase in `V`. Synapse-type `gf`, an exponential increase.

| Synapse Type | Behavior |
|--------------|----------|
| `V`          | Instantaneous jump in membrane potential `V`, potentially emitting spike |
| `ge`         | Slow, steady increase in `V` over time |
| `gf + gate`  | Fast, nonlinear voltage rise due to exponential dynamics |
| `gate`       | Controls whether `gf` affects the neuron at all |

###  Event-by-event explanation of the animation

| Time (ms) | Type    | Value | Description |
|-----------|---------|-------|-------------|
| `t = 20`  | `V`     | 10.0 (`=Vt`)  | Instantaneously pushes `V` to threshold, triggering a spike |
| `t = 60`  | `ge`    | 2.0   | Applies constant integration current: slow, linear voltage increase |
| `t = 100` | `gf`    | 2.5   | Adds fast-decaying input, gated via `gate = 1` at same time |
| `t = 160` | `V`     | 2.0   | Small, instant boost to `V` |
| `t = 200` | `gate`  | -1.0  | Disables exponential decay pathway by zeroing the gate signal |

---

###  `t = 20 ms - V(10.0)`
- A **V-synapse** adds +10.0 mV to `V` instantly.
- Since `Vt = 10.0`, this causes **immediate spike**.
- The neuron resets: `V → 0`, `ge, gf, gate → 0`.

**Effect**: Demonstrates a direct spike trigger via instantaneous voltage jump.

---

###  `t = 60 ms — ge(2.0)`
- A **ge-synapse** applies constant input current.
- Voltage rises **linearly** over time.
- Alone, this isn't sufficient to reach `Vt`, so no spike occurs yet.

**Effect**: Shows the smooth effect of continuous integration from ge-type input.

---

###  `t = 100 ms — gf(2.5)` and `gate(1.0)`
- A **gf-synapse** delivers fast-decaying input current.
- A **gate-synapse** opens the gate (`gate = 1`), activating `gf` dynamics.
- Voltage rises **nonlinearly** as `gf` initially dominates, then decays.
- Combined effect from earlier `ge` and `gf` **causes a spike** shortly after.

**Effect**: Demonstrates exponential integration (gf) gated for a temporary burst.

---

###  `t = 160 ms — V(2.0)`
- A small **V-synapse** bump of +2.0 mV occurs.
- This is **not enough** to cause a spike, but it shifts `V` upward instantly.

**Effect**: Shows subthreshold perturbation from a V-type synapse.


###  `t = 200 ms — gate(-1.0)`
- The **gate is closed** (`gate = 0`), disabling `gf` decay term.
- Any remaining `gf` is no longer integrated into `V`.

**Effect**: Demonstrates control logic: `gf` is disabled, computation halts.

## Benefits of This Model

This neuron model is designed for **interval-coded** values. Time **intervals between spikes** directly encode numeric values.

The neuron model has dynamic behaviours that eenable symbolic operations such as memory, arithmetic, and differential equation solving. The dynamics of this neuron model forms a **Turing-complete** computation framework (for in depth information, refer to the [STICK paper](https://arxiv.org/abs/1507.06222)).

This neuron model has the following characteristics:

- **Compact**: Minimal neurons required for functional blocks
- **Precise**: Accurate sub-millisecond spike-based encoding
- **Composable**: Modular design supports hierarchical circuits
- **Hardware-Compatible**: Ported to digital integrate-and-fire cores