# STICK emulator

Python package to emulate the networks implemented in [STICK](https://arxiv.org/abs/1507.06222) (Spike Time Interval Computation Kernel)

## Installation
```
cd stick-emulator
pip install -e .
pip install -r requirements.txt
```

> Note: Use the following flag if you're using VSCode and you want it to see the package, installed in editable mode.
> `pip install -e . --config-settings editable_mode=compat`

## Example of use

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
Visualization tools are available to inspect the network topology and the spiking chronogram. They can be triggered with `VIS=1` without modifying the source code.

```bash
VIS=1 python simulate.py
```

> **Note on topology visualization:**
> If a module contains submodules, submodules are displayed as boxes whose content is hidden. The top module neurons are shown together with the submodule neurons which are directly connected to the top module ones.

**Example:** Visualizing a multiplier network

<img width="1430" alt="Screenshot 2025-05-01 at 16 39 11" src="https://github.com/user-attachments/assets/cf9e18c5-d496-4f9b-979d-15f02ba230dd" />
<img width="1078" alt="Screenshot 2025-05-01 at 16 40 23" src="https://github.com/user-attachments/assets/192040a4-021b-488f-8c8f-fca75e039a08" />
