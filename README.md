# STICK emulator

Python package to emulate the networks implemented in [STICK](https://arxiv.org/abs/1507.06222) (Spike Time Interval Computation Kernel)

## Installation
```
cd stick-emulator
pip install -e .
```

## Use

1. Define your network

```python
class class TestModule(SpikingNetworkModule):
    def __init__(self, encoder, module_name):
        super().__init__(module_name)

        Vt = 10.0
        tm = 100.0
        tf = 20.0
        Tsyn = 1.0

        self.inp = self.add_neuron(Vt, tm, tf, Vreset=0, neuron_name='input')
        self.outp = self.add_neuron(Vt, tm, tf, Vreset=0, neuron_name='output')

        self.connect_neurons(self.inp, self.outp, "ge", weight=2*we, delay=Tsyn)
        self.connect_neurons(self.inp, self.outp, "gf", weight=2*we, delay=Tsyn)
```

2. Use the simulator on your network
```python
# simulate.py
encoder = DataEncoder()
net = TestModule(encoder, 'testnet')

inp_val = 0.6
sim = Simulator(net, encoder, dt=0.01)
sim.apply_input_value(inp_val, neuron=net.inp, t0=5)
sim.simulate(300)
```

3. Run your network
```bash
python simulate.py
```

## Network visualization
Topology visualization is available to help inspect and debug your network.

```bash
VIS=1 python simulate.py
```

> **Note:**
> If a module contains submodules, only the neurons (and synapses) of the submodule that interact with the neurons in the main module are displayed.

