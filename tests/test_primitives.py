# Copyright (C) 2025  Neucom Aps
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pytest
from axon_sdk.primitives import SpikingNetworkModule, ExplicitNeuron


def test_basic_module():
    class MockModule(SpikingNetworkModule):
        def __init__(self):
            super().__init__()
            self.neuron = self.add_neuron(Vt=0, tm=0, tf=0, neuron_name="neuron1")

    module = MockModule()

    assert len(module.neurons) == 1, "module.neurons should contain 1 neuron"
    assert isinstance(module.neurons, list), "module.neurons should be a list"


def test_deep_module():
    class MockModule(SpikingNetworkModule):
        def __init__(self):
            super().__init__()
            self.add_neuron(Vt=0, tm=0, tf=0, neuron_name="neuron1")

    class WrapModule(SpikingNetworkModule):
        def __init__(self):
            super().__init__()
            mock_module = MockModule()
            self.add_neuron(Vt=0, tm=0, tf=0, neuron_name="neuron2")
            self.add_subnetwork(mock_module)

    module = WrapModule()

    assert len(module.neurons) == 2, "module.neurons should contain 2 neurons"
    assert isinstance(module.neurons, list), "module.neurons should be a list"


def test_deeper_module():
    class MockModule(SpikingNetworkModule):
        def __init__(self):
            super().__init__()
            self.add_neuron(Vt=0, tm=0, tf=0, neuron_name="neuron1")

    class Wrap1Module(SpikingNetworkModule):
        def __init__(self):
            super().__init__()
            mock_module = MockModule()
            self.add_neuron(Vt=0, tm=0, tf=0, neuron_name="neuron2")
            self.add_subnetwork(mock_module)

    class Wrap2Module(SpikingNetworkModule):
        def __init__(self):
            super().__init__()
            wrap_module = Wrap1Module()
            self.add_neuron(Vt=0, tm=0, tf=0, neuron_name="neuron3")
            self.add_subnetwork(wrap_module)

    module = Wrap2Module()

    assert len(module.neurons) == 3, "module.neurons should contain 3 neurons"
    assert isinstance(module.neurons, list), "module.neurons should be a list"
