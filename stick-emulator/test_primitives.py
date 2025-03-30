import pytest

from ..primitives import SpikingNetworkModule, ExplicitNeuron


def test_basic_module():
    class MockModule(SpikingNetworkModule):
        def __init__(self):
            super().__init__()
            neuron = ExplicitNeuron(Vt=0, tm=0, tf=0, neuron_id='neuron1')
            self.add_neuron(neuron)

    module = MockModule()

    assert len(module.neurons) == 1, "module.neurons should contain 1 neuron"
    assert isinstance(module.neurons, list), "module.neurons should be a list"
    assert 'neuron1' in module.neurons, "neuron should be in module.neurons"    


def test_deep_module():
    class MockModule(SpikingNetworkModule):
        def __init__(self):
            super().__init__()
            neuron = ExplicitNeuron(Vt=0, tm=0, tf=0, neuron_id='neuron1')
            self.add_neuron(neuron)

    class WrapModule(SpikingNetworkModule):
        def __init__(self):
            super().__init__()
            mock_module = MockModule()
            neuron = ExplicitNeuron(Vt=0, tm=0, tf=0, neuron_id='neuron2')
            self.add_neuron(neuron)
            self.add_subnetwork(mock_module)

    module = WrapModule()

    assert len(module.neurons) == 2, "module.neurons should contain 2 neurons"
    assert isinstance(module.neurons, list), "module.neurons should be a list"
    assert 'neuron1' in module.neurons, "neuron should be in module.neurons"
    assert 'neuron2' in module.neurons, "neuron should be in module.neurons"


def test_deeper_module():
    class MockModule(SpikingNetworkModule):
        def __init__(self):
            super().__init__()
            neuron = ExplicitNeuron(Vt=0, tm=0, tf=0, neuron_id='neuron1')
            self.add_neuron(neuron)

    class Wrap1Module(SpikingNetworkModule):
        def __init__(self):
            super().__init__()
            mock_module = MockModule()
            neuron = ExplicitNeuron(Vt=0, tm=0, tf=0, neuron_id='neuron2')

            self.add_neuron(neuron)
            self.add_subnetwork(mock_module)

    class Wrap2Module(SpikingNetworkModule):
        def __init__(self):
            super().__init__()
            wrap_module = Wrap1Module()
            neuron = ExplicitNeuron(Vt=0, tm=0, tf=0, neuron_id='neuron3')

            self.add_neuron(neuron)
            self.add_subnetwork(wrap_module)

    module = Wrap2Module()

    assert len(module.neurons) == 3, "module.neurons should contain 3 neurons"
    assert isinstance(module.neurons, list), "module.neurons should be a list"
    assert 'neuron1' in module.neurons, "neuron should be in module.neurons"
    assert 'neuron2' in module.neurons, "neuron should be in module.neurons"
    assert 'neuron3' in module.neurons, "neuron should be in module.neurons"
    


