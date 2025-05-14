from stick_emulator.primitives import (
    SpikingNetworkModule,
    DataEncoder,
    SpikeEventQueue,
    ExplicitNeuron,
)

from stick_emulator.visualization import vis_topology, plot_chronogram
import os


class Simulator:
    def __init__(
        self, net: SpikingNetworkModule, encoder: DataEncoder, dt: float = 0.001
    ) -> None:
        self.net = net
        self.event_queue = SpikeEventQueue()
        self.encoder = encoder
        self.dt = dt
        self.timesteps: list[float] = []

        # Optimization: Pre-fetch the list of all neurons.
        # This will use the cached version in SpikingNetworkModule after first call.
        self._all_neurons: list[ExplicitNeuron] = self.net.neurons

        # CORRECTED: Initialize logs here to ensure they are ready before simulate()
        # or if apply_input_value/spike is called before simulate.
        self.spike_log: dict[str, list[float]] = {
            neuron.uid: [] for neuron in self._all_neurons
        }
        self.voltage_log: dict[str, list[float]] = {
            neuron.uid: [] for neuron in self._all_neurons
        }

    def apply_input_value(
        self, value: float, neuron: ExplicitNeuron, t0: float = 0
    ):
        assert value >= 0.0 and value <= 1.0
        # Ensure the input neuron is known to the simulator's logs
        if neuron.uid not in self.spike_log:
            # This case should ideally not happen if neuron is part of self.net
            # and __init__ correctly initialized logs.
            # However, as a safeguard or for dynamically added neurons (not current design):
            self.spike_log[neuron.uid] = []
            # self.voltage_log[neuron.uid] = [] # if apply_input_value could also affect voltage log directly

        spike_interval = self.encoder.encode_value(value)
        for t_spike_in_interval in spike_interval:
            event_time = t0 + t_spike_in_interval
            self._log_spike_occurrence(neuron=neuron, t=event_time)
            for synapse in neuron.out_synapses:
                self.event_queue.add_event(
                    time=event_time + synapse.delay,
                    neuron=synapse.post_neuron,
                    synapse_type=synapse.type,
                    weight=synapse.weight,
                )

    def apply_input_spike(self, neuron: ExplicitNeuron, t: float):
        # Ensure the input neuron is known to the simulator's logs
        if neuron.uid not in self.spike_log:
            self.spike_log[neuron.uid] = []

        self._log_spike_occurrence(neuron, t)
        for synapse in neuron.out_synapses:
            self.event_queue.add_event(
                time=t + synapse.delay,
                neuron=synapse.post_neuron,
                synapse_type=synapse.type,
                weight=synapse.weight,
            )

    def simulate(self, simulation_time: float):
        num_steps = int(simulation_time / self.dt)
        self.timesteps = [(i + 1) * self.dt for i in range(num_steps)]

        # Logs are now initialized in __init__.
        # If there's a need to clear logs for a new simulation run with the same Simulator instance:
        # for uid in self.spike_log: self.spike_log[uid].clear()
        # for uid in self.voltage_log: self.voltage_log[uid].clear()
        # However, typically a new Simulator instance or a dedicated reset method would be used.
        # For this problem, we assume simulate() is called once or state is cumulative.

        for t in self.timesteps:
            events = self.event_queue.pop_events(t)

            affected_neurons_in_timestep = set()
            for event in events:
                event.affected_neuron.receive_synaptic_event(
                    event.synapse_type, event.weight
                )
                affected_neurons_in_timestep.add(event.affected_neuron)

            for neuron in self._all_neurons:
                needs_update = (
                    neuron in affected_neurons_in_timestep
                    or neuron.ge != 0.0
                    or neuron.gf != 0.0
                    or neuron.gate != 0
                )

                current_V = neuron.V
                if needs_update:
                    (V_updated, spike) = neuron.update_and_spike(self.dt)
                    current_V = V_updated
                    if spike:
                        self._log_spike_occurrence(neuron=neuron, t=t)
                        neuron.reset()
                        for synapse in neuron.out_synapses:
                            self.event_queue.add_event(
                                time=t + synapse.delay,
                                neuron=synapse.post_neuron,
                                synapse_type=synapse.type,
                                weight=synapse.weight,
                            )

                self._log_voltage_value(neuron=neuron, V=current_V)

        if os.getenv("VIS", "0") == "1":
            self.launch_visualization()

    def _log_spike_occurrence(self, neuron: ExplicitNeuron, t: float) -> None:
        # Assumes neuron.uid is a key in self.spike_log due to __init__ pre-initialization.
        self.spike_log[neuron.uid].append(t)

    def _log_voltage_value(self, neuron: ExplicitNeuron, V: float) -> None:
        # Assumes neuron.uid is a key in self.voltage_log due to __init__ pre-initialization.
        self.voltage_log[neuron.uid].append(V)

    def launch_visualization(self):
        vis_topology(self.net)
        plot_chronogram(
            timesteps=self.timesteps,
            voltage_log=self.voltage_log,
            spike_log=self.spike_log,
        )
