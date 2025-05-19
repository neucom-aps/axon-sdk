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
