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

    def __lt__(self, other):  # Needed to use it in a heap
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


class EventQueue:
    def __init__(self):
        self.events = []

    def add_event(self, event_time, neuron_uid, synapse_type, weight):
        heapq.heappush(self.events, (event_time, neuron_uid, synapse_type, weight))

    def pop_events(self, current_time):
        events = []
        while self.events and self.events[0][0] <= current_time:
            events.append(heapq.heappop(self.events))
        return events
