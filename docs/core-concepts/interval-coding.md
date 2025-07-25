# Interval Coding


Axon encodes scalars values into pairs of spikes. The time difference between the 2 spikes in a pair, also called the  **Inter Spike Interval (ISI)**, relates to the encoded value. The `DataEncoder` class is responsible for encoding values into spikes and decoding them back.

This document highlight how this encoding and decoding process works.

scalar values into **inter-spike intervals (ISIs)** and decoding them back. This functionality is central to how Axon implements symbolic computation using the **STICK (Spike Time Interval Computational Kernel)** framework.

This document explains how Axon implements the **interval-based encoding and computation** as defined by the STICK (Spike Time Interval Computational Kernel) model.

---


##  Concept

In Axon, numerical values are encoded in the **time difference between two spikes**. Other neuromorphic models use encoding schemes such as spike rates or voltage values. However, time-encoding has the following advantages:

- Keeps the sparsity of spike events and hence enables ultra-low power consumption.
- Allows for high numeric resolution by using time domain while keeping spikes as binary events.

![ISI encoding](../figs/isi.png)


Values **x ∈ [0,1]** are encoded in the time difference Δt between two spikes:

```text
ΔT = Tmin + x * Tcod
x   = (ΔT − Tmin) / Tcod
```

where:
- **`Tmin`**: minimum time difference
- **`Tcod`**: coding interval
- **`ΔT`**: time difference between two spikes

As a consequence, `Tmax = Tmin + Tcod`.

Default coding values are:

| Parameter | Numeric value (ms)    | Meaning                        |
|-----------|----------|--------------------------------|
| `Tmin`      | 10.0  | Spiking threshold              |
| `Tcod`  | 100.0  | Voltage after reset            |
| `Tmax`      | 110.0 | Membrane integration constant  |

**Example**:
```text
ΔT(0.6) = 10 ms + 0.6 * 100 ms = 70 ms
```


## Interval-based Computation
Spiking networks are built to process interval-encoded values. The network dynamics are governed by the synaptic weights and delays, allowing for complex computations based on the timing of spikes.
* Value `x` is represented by timing between spikes Δt.
* Spiking networks manipulate these intervals via synaptic delays, integration, and gating, executing operations like addition, multiplication, and memory.


##  `DataEncoder` Class
The `DataEncoder` class provides methods for encoding and decoding values:

```python
class DataEncoder:
    def __init__(self, Tmin=10.0, Tcod=100.0):
        ...

    def encode_value(self, value: float) -> tuple[float, float]:
        ...

    def decode_interval(self, spiking_interval: float) -> float:
        ...
```

The `DataEncoder` is used during simulation and output processing:

```python
from axon_sdk.simulator import Simulator
from axon_sdk.encoders import DataEncoder

encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
spike_pair = encoder.encode_value(0.6)
```

```text
spike_pair
>> (0.0, 70.0)
```

![Encoder](../figs/isi_encoder.png)

This spike pair can then be injected into the network, and the simulator will handle the timing based on the encoded intervals.

```python
sim.apply_input_value(value=0.6, neuron=input_neuron)
```

## Output Decoding
```python
interval = spike_pair[1] - spike_pair[0]
decoded_value = encoder.decode_interval(interval)
```

```text
decoded_value
>> 0.6
```

