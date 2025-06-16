from .helpers import flatten_nested_list

from typing import Optional


class AbstractNeuron:
    _instance_count = 0

    def __init__(
        self,
        Vt,
        tm,
        tf,
        Vreset=0.0,
        neuron_name: Optional[str] = None,
        parent_mod_id: Optional[str] = None,
        additional_info: Optional[str] = None
    ):
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

        if parent_mod_id is not None:
            self._uid = (
                f"(m{parent_mod_id},n{AbstractNeuron._instance_count})_{neuron_name}"
            )
        else:
            self._uid = f"(n{AbstractNeuron._instance_count})_{neuron_name}"

        self.additional_info = additional_info
        AbstractNeuron._instance_count += 1

    @property
    def uid(self) -> str:
        return self._uid

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
            self.gate += weight
        else:
            raise ValueError("Unknown synapse type.")


class ExplicitNeuron(AbstractNeuron):
    def __init__(
        self,
        Vt: float,
        tm: float,
        tf: float,
        Vreset: float = 0.0,
        neuron_name: Optional[str] = None,
        parent_mod_id: Optional[str] = None,
        additional_info: Optional[str] = None
    ):
        super().__init__(Vt, tm, tf, Vreset, neuron_name, parent_mod_id, additional_info)
        self.spike_times: list[float] = []
        self.out_synapses: list[Synapse] = []

    def reset(self):
        self.V = self.Vreset
        self.ge = 0
        self.gf = 0
        self.gate = 0


class Synapse:
    _instance_count = 0

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

        self._uid = f"synapse_{Synapse._instance_count}"
        Synapse._instance_count += 1

    @property
    def uid(self) -> str:
        return self._uid
