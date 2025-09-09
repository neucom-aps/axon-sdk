<p align="center">
  <img src="https://github.com/user-attachments/assets/3c445fa2-ea61-4bb2-a625-af79d46751e6" alt="Sublime's custom image"/>
</p>

**Axon SDK** is a Python toolkit for building and simulating Spiking Neural Networks (SNN) based on the [STICK framework](https://arxiv.org/abs/1507.06222) (Spike Time Interval Computation Kernel).

It provides a modular, event-driven framework for developing symbolic, time-coded spiking algorithms for control, computation and embedded neuromorphic applications. 

## Features
-  **Library of SNN computation kernels**
-  **Accurate simulation** of inter-spike-interval coding
-  **Structured APIs** for network composition and module extension
-  **Interactive visualization** of spike chronograms and topologies

## Installation
```bash
cd axon-sdk
pip install -e .
pip install -r requirements.txt
```

> Note: Installing the package with the following flag will allow interactive code completion in VSCode:
> 
> `pip install -e . --config-settings editable_mode=compat`

## Documentation and tutorials
[neucom-aps.github.io/axon-sdk/](https://neucom-aps.github.io/axon-sdk/)


## Example: Define custom network

1. Define your network

```python
class TestModule(SpikingNetworkModule):
    def __init__(self, encoder):
        super().__init__()

        Vt = 10.0
        tm = 100.0
        tf = 20.0
        Tsyn = 1.0
        wacc = (Vt * tm) / encoder.Tmax

        self.input = self.add_neuron(Vt, tm, tf, Vreset=0, neuron_name='input')
        self.middle = self.add_neuron(Vt, tm, tf, Vreset=0, neuron_name='middle')
        self.output = self.add_neuron(Vt, tm, tf, Vreset=0, neuron_name='output')

        self.connect_neurons(self.input, self.middle, "ge", weight=wacc, delay=Tsyn)
        self.connect_neurons(self.middle, self.output, "ge", weight=wacc, delay=Tsyn)
```

2. Simulate your network
```python
# simulate.py
encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
net = TestModule(encoder)

sim = Simulator(net, encoder, dt=0.01)
sim.apply_input_spike(neuron=net.input, t=5)
sim.simulate(300)
```

3. Extract output spikes
```python
out_spikes = sim.sim.spike_log[net.outp.uid]
```

## Example: Use predefined computation networks
We provide a library of predefined computation modules:

```python
from axon_sdk.networks import MultiplierNetwork
from axon_sdk.primitives import DataEncoder
from axon_sdk.simulator import Simulator

encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
net = MultiplierNetwork(encoder)

spikes = encoder.encode_value(0.2)
interval = spikes[1] - spikes[0]
value = encoder.decode_interval(interval)

sim = Simulator(net, encoder, dt=0.01)
sim.apply_input_value(0.9, neuron=net.input1)
sim.apply_input_value(0.8, neuron=net.input2)
sim.simulate(simulation_time=400)
spikes = sim.spike_log.get(net.output.uid, [])
interval = spikes[1] - spikes[0]
```

```python
output_val = encoder.decode_interval(interval)
>> 0.72
```

![mul-snn-linkedin](https://github.com/user-attachments/assets/d99117f9-fe1a-4237-9f81-b1feaf0ed329)

## License
**Axon SDK** is published under the GPLv3 license, which applies to all files in this repository.

<p align="right">
  <img src="https://komarev.com/ghpvc/?username=neucom-aps&color=orange&style=pixel&label=VISITOR+COUNT" alt=”tomkaX” />
</p>



