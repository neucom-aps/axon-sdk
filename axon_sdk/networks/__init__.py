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

# Memory networks
from .memory.memory import MemoryNetwork
from .memory.inverting_memory import InvertingMemoryNetwork
from .memory.signed_memory import SignedMemoryNetwork
from .memory.constant import ConstantNetwork
from .memory.signed_constant import SignedConstantNetwork

# Connecting networks
from .connecting.synchronizer import SynchronizerNetwork

# Functional networks
from .functional.subtractor import SubtractorNetwork
from .functional.linear_combinator import LinearCombinatorNetwork
from .functional.exponential import ExponentialNetwork
from .functional.natural_log import LogNetwork
from .functional.multiplier import MultiplierNetwork
from .functional.signed_multiplier import SignedMultiplierNetwork
from .functional.scalar_multiplier import ScalarMultiplierNetwork
from .functional.divider import DivNetwork
from .functional.adder import AdderNetwork
from .functional.signflip import SignFlipperNetwork
from .functional.signed_multipler_norm import SignedMultiplierNormNetwork
