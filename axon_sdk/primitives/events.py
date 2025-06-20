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

import heapq
from .elements import ExplicitNeuron


class SpikeEvent:
    def __init__(
        self,
        time: float,
        affected_neuron: ExplicitNeuron,
        synapse_type: str,
        weight: float,
    ):
        self.time = time
        self.affected_neuron = affected_neuron
        self.synapse_type = synapse_type
        self.weight = weight

    def __lt__(self, other):
        # Needed since spike events will be used in a heap
        return self.time < other.time


class SpikeEventQueue:
    def __init__(self):
        self.events: list[SpikeEvent] = []

    def add_event(
        self,
        time: float,
        neuron: ExplicitNeuron,
        synapse_type: str,
        weight: float,
    ):
        event = SpikeEvent(
            time=time,
            affected_neuron=neuron,
            synapse_type=synapse_type,
            weight=weight,
        )
        heapq.heappush(self.events, event)

    def pop_events(self, current_time) -> list[SpikeEvent]:
        events = []
        while self.events and self.events[0].time <= current_time:
            events.append(heapq.heappop(self.events))
        return events
