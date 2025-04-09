from abc import ABC, abstractmethod
from collections import defaultdict
import uuid
import random

from typing import Optional


class AbstractNeuron:
    def __init__(self, Vt, tm, tf, Vreset=0.0):
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

    def update(self, dt) -> bool:
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

        # Check for spike condition (reset happens explicitly later)
        if self.V >= self.Vt:
            return True
        return False

    def update_and_spike(self, dt) -> tuple[float, bool]:
        """
        Update the state of the neuron by one timestep.

        Parameters:
        dt (float): The simulation timestep (ms).

        Returns:
        bool: True if the neuron spikes, False otherwise.
        """
        spike = False
        # Update membrane potential based on the differential equations provided
        self.V += dt * (self.ge + self.gate * self.gf) / self.tm

        # Update gf dynamics if gated
        if self.gate:
            self.gf -= dt * (self.gf / self.tf)

        # Check for spike condition (reset happens explicitly later)
        if self.V >= self.Vt:
            spike = True
        return (self.V, spike)

    def receive_synaptic_event(self, synapse_type, weight):
        """
        Update neuron state based on incoming synaptic event.

        Parameters:
        synapse_type (str): Type of synapse ('V', 'ge', 'gf', 'gate').
        weight (float): Synaptic weight to modify neuron state.
        """
        if synapse_type == "V":
            self.V += weight
        elif synapse_type == "ge":
            self.ge += weight
        elif synapse_type == "gf":
            self.gf += weight
        elif synapse_type == "gate":
            self.gate = 1 if weight > 0 else 0
        else:
            raise ValueError("Unknown synapse type.")


class ExplicitNeuron(AbstractNeuron):
    def __init__(
        self,
        Vt: float,
        tm: float,
        tf: float,
        Vreset: float = 0.0,
        neuron_id: Optional[str] = None,
    ):
        super().__init__(Vt, tm, tf, Vreset)
        self.id = neuron_id
        self.spike_times: list[float] = []
        self.out_synapses: list[Synapse] = []

        if not neuron_id:
            vowel = ["a", "e", "i", "o", "u"]
            consonant = [
                "b",
                "c",
                "d",
                "f",
                "g",
                "h",
                "j",
                "k",
                "l",
                "m",
                "n",
                "p",
                "q",
                "r",
                "s",
                "t",
                "v",
                "w",
                "x",
                "y",
                "z",
            ]
            choose = lambda x: random.choice(x)
            neuron_id = f"{choose(consonant) + choose(vowel) + choose(consonant) + choose(vowel)} (random name)"
            self.id = neuron_id
        self.id = neuron_id
        self.uid = uuid.uuid4()

    def reset(self):
        self.V = self.Vreset
        self.ge = 0
        self.gf = 0
        self.gate = 0


class Synapse:
    def __init__(
        self,
        pre_neuron: ExplicitNeuron,
        post_neuron: ExplicitNeuron,
        weight: float,
        delay: float,
        synapse_type: str,
    ):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.type = synapse_type
        self.weight = weight
        self.delay = delay
