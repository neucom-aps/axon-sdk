import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from collections import defaultdict
import heapq


class AbstractNeuron:
    def __init__(self, Vt=10.0, Vreset=0.0, tm=100.0, tf=20.0):
        """
        Initialize the neuron with given parameters.

        Parameters:
        Vt (float): Spike threshold voltage (mV).
        Vreset (float): Reset voltage after spike (mV).
        tm (float): Membrane time constant (ms).
        tf (float): Time constant for exponential synapses (ms).
        """
        self.Vt = Vt
        self.Vreset = Vreset
        self.tm = tm
        self.tf = tf

        # Initialize state variables
        self.V = Vreset
        self.ge = 0.0
        self.gf = 0.0
        self.gate = 0

    def update(self, dt):
        """
        Update the state of the neuron by one timestep.

        Parameters:
        dt (float): The simulation timestep (ms).

        Returns:
        bool: True if the neuron spikes, False otherwise.
        """
        # Update membrane potential based on the differential equations provided
        self.V += dt * (self.ge + self.gate * self.gf) / self.tm

        # Update gf dynamics if gated
        if self.gate:
            self.gf -= dt * (self.gf / self.tf)

        # Check for spike condition
        if self.V >= self.Vt:
            self.V = self.Vreset
            self.ge = 0.0
            self.gf = 0.0
            self.gate = 0
            return True
        return False

    def receive_synaptic_event(self, synapse_type, weight):
        """
        Update neuron state based on incoming synaptic event.

        Parameters:
        synapse_type (str): Type of synapse ('V', 'ge', 'gf', 'gate').
        weight (float): Synaptic weight to modify neuron state.
        """
        if synapse_type == 'V':
            self.V += weight
        elif synapse_type == 'ge':
            self.ge += weight
        elif synapse_type == 'gf':
            self.gf += weight
        elif synapse_type == 'gate':
            self.gate = 1 if weight > 0 else 0
        else:
            raise ValueError("Unknown synapse type.")
        
class ExplicitNeuron(AbstractNeuron):
    def __init__(self, neuron_id, Vt=10, Vreset=0, tm=100, tf=20):
        super().__init__(Vt, Vreset, tm, tf)
        self.id = neuron_id
        self.spike_times = []

    def reset(self):
        self.V = self.Vreset
        self.ge = 0
        self.gf = 0
        self.gate = 0
        

class DataEncoder:
    def __init__(self, Tmin=10.0, Tcod=100.0):
        """
        Initialize the encoder with the given minimum interval and coding time.

        Parameters:
        Tmin (float): Minimum spike interval (ms).
        Tcod (float): Interval duration representing the maximum encoded value (ms).
        """
        self.Tmin = Tmin
        self.Tcod = Tcod

    def encode_value(self, value):
        """
        Encode a value into spike times.

        Parameters:
        value (float): The value to encode, expected between 0 and 1.

        Returns:
        tuple: Two spike times representing the encoded value.
        """
        interval = self.Tmin + value * self.Tcod
        return (0, interval)
    
class AbstractSpikingNetwork(ABC):
    def __init__(self):
        self.neurons = {}
        self.connections = defaultdict(dict)

    def add_neuron(self, neuron_id, neuron):
        self.neurons[neuron_id] = neuron

    def connect_neurons(self, pre_neuron_id, post_neuron_id, synapse_type, weight_delay_tuple):
        self.connections[pre_neuron_id][post_neuron_id] = (synapse_type, weight_delay_tuple)

    @abstractmethod
    def simulate(self, simulation_time, dt):
        pass

class Synapse:
    def __init__(self, pre_neuron, post_neuron, synapse_type, weight, delay):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.type = synapse_type
        self.weight = weight
        self.delay = delay


class EventQueue:
    def __init__(self):
        self.events = []

    def add_event(self, event_time, neuron_id, synapse_type, weight):
        heapq.heappush(self.events, (event_time, neuron_id, synapse_type, weight))

    def pop_events(self, current_time):
        events = []
        while self.events and self.events[0][0] <= current_time:
            events.append(heapq.heappop(self.events))
        return events