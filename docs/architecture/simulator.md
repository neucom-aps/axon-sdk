# Axon Simulator Engine

The Axon simulator executes symbolic spiking neural networks (SNNs) built with the STICK (Spike Time Interval Computational Kernel) model. This document describes the simulation engine's architecture, parameters, workflow, and features.

---

##  1. Purpose

The `Simulator` class provides a discrete-time, event-driven environment to simulate:

- Spiking neuron dynamics
- Synaptic event propagation
- Interval-based input encoding and output decoding
- Internal logging of voltages and spikes

It is optimized for symbolic, low-rate temporal computation rather than high-frequency biological modeling.

---

##  2. Core Components

| Component          | Description |
|-------------------|-------------|
| `net`             | The user-defined spiking network (a `SpikingNetworkModule`) |
| `dt`              | Simulation timestep in seconds (default: `0.001`) |
| `event_queue`     | Priority queue managing scheduled synaptic events |
| `encoder`         | Object for encoding/decoding interval-coded values |
| `spike_log`       | Maps neuron UIDs to their spike timestamps |
| `voltage_log`     | Records membrane voltage per neuron per timestep |

---

##  3. Simulation Loop

The simulator proceeds in `dt`-sized increments for a specified duration:

1. **Event Queue Check**  
   All scheduled events due at time `t` are popped.

2. **Synaptic Updates**  
   Each event updates the target neuron's state (`V`, `ge`, `gf`, or `gate`).

3. **Neuron Updates**  
   Each affected neuron is numerically integrated using:
```python
    V += (ge + gate * gf) * dt / tau_m
    ```
    where `tau_m` is the membrane time constant.    




4. **Spike Detection & Reset**  
If `V ≥ Vt`, the neuron spikes:
- `V ← Vreset`, `ge ← 0`, `gf ← 0`, `gate ← 0`
- All outgoing synapses generate future spike events

5. **Activity Tracking**  
Neurons with non-zero `ge`, `gf`, or `gate` are marked active for the next step.

---

## 🎚 4. Configuration Knobs

| Parameter | Description | Default |
|----------|-------------|---------|
| `dt`     | Time resolution per step (in seconds) | `0.001` (1 ms) |
| `Tmin`   | Minimum interspike delay in encoding | `10.0` ms |
| `Tcod`   | Encoding range above Tmin | `100.0` ms |
| `simulation_time` | Total simulation duration (in seconds) | user-defined |

These settings are defined at the simulator or encoder level depending on purpose.

---

## 🔌 5. Inputs & Injection

### `apply_input_value(value, neuron, t0=0)`
Injects a scalar `value ∈ [0, 1]` into a neuron via interval-coded spike pair.

### `apply_input_spike(neuron, t)`
Injects a single spike into a neuron at exact time `t`.

---

## 📤 6. Output Decoding

To read results from signed STICK outputs:

```python
from axon.simulator import decode_output

value = decode_output(simulator, reader)
```
Decodes interval between two spikes on either the + or − output neuron

Returns a signed scalar in [−1, 1] (scaled by `reader.normalization`)

## 7. Loggin and Visualization
The simulator maintains:
* `spike_log`: Maps neuron UIDs to spike timestamps with: {neuron_uid: [t0, t1, ...]}
* `voltage_log`: Maps neuron UIDs to their membrane voltages at each timestep with: {neuron_uid: [V0, V1, ...]}

Optional visualization can be enabled by setting `VIS=1` in your environment. 
```python
sim.launch_visualization()
```

* `plot_chronogram()`: Spike raster and voltage traces
* `vis_topology()`: Interactive network topology visualization 

## 8. Design Flow
1. **Define Network**: Create a `SpikingNetworkModule` with neurons and synapses.
```python
from axon_sdk.network import SpikingNetworkModule
net = SpikingNetworkModule()
```
2. **Instantiate Encoder**: Create an encoder for interval coding.
```python
from axon_sdk.encoder import DataEncoder
encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
```


3. **Instantiate Simulator**: Create a `Simulator` instance with the network and parameters.
```python
sim = Simulator(net, encoder dt=0.001)
```
4. **Apply Inputs**: Use `apply_input_value()` or `apply_input_spike()` to inject data.
```python
sim.apply_input_value(0.5, neuron_uid, t0=0)
```

5. **Run Simulation**: Execute the simulation for a specified duration of timesteps.
```python
sim.run(simulation_time=100)
```

6. **Analyze Outputs**: Use `decode_output()` to read results from the simulation.
```python
value = sim.decode_output(reader)
```

## Example Usage
```python
from axon_sdk.simulator import Simulator
from axon_sdk.networks import MultiplierNetwork
from axon_sdk.utils import encode_interval

net = MultiplierNetwork()
encoder = DataEncoder()
sim = Simulator(net, encoder, dt=0.001)

a, b = 0.4, 0.25
sim.apply_input_value(a, net.input1)
sim.apply_input_value(b, net.input2)
sim.simulate(simulation_time=0.5)
sim.plot_chronogram()
```
## 11. Summary
* Event-driven, millisecond-resolution simulator

* Supports interval-coded STICK networks

* Accurate logging of all internal neuron dynamics

* Integrates seamlessly with compiler/runtime interfaces


