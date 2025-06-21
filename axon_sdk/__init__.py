"""
Axon SDK
========

Axon is a Python-based simulation toolkit for spike-timing-based computation using the STICK (Spike Timing Interval Computational Kernel) model.

It provides:
- Primitives for defining STICK neurons and synapses.
- Tools for composing symbolic spiking networks.
- Compilation routines to map symbolic models to executable primitives.
- A simulator for event-driven execution and visualization of spike dynamics.

This SDK is part of the Neucom platform developed by Neucom ApS.
"""

# Axon SDK — A simulation framework for spike-timing-based neural computation using the STICK model.
# Copyright (C) 2024–2025 Neucom ApS
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

from .simulator import Simulator
