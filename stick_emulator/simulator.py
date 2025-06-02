from stick_emulator.primitives import (
    SpikingNetworkModule,
    DataEncoder,
    SpikeEventQueue,
    ExplicitNeuron,
)
from stick_emulator.visualization import vis_topology, plot_chronogram
from stick_emulator.compiler import ExecutionPlan
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
        # Cache the neurons in the network to avoid continouous list comprehension
        self._cached_neurons: list[ExplicitNeuron] = self.net.neurons
        self.spike_log: dict[str, list[float]] = {}
        self.voltage_log: dict[str, list[tuple]] = {}
        for neuron in self._cached_neurons:
            self.spike_log[neuron.uid] = []
            self.voltage_log[neuron.uid] = []

    def apply_input_value(self, value: float, neuron: ExplicitNeuron, t0: float = 0):
        assert value >= 0.0 and value <= 1.0

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
        # Set to track neurons with non-zero ge, gf, or gate at the end of a timestep
        active_state_neurons: set[ExplicitNeuron] = set()

        for i, t in enumerate(self.timesteps):
            events = self.event_queue.pop_events(t)

            currently_affected_neurons = set()
            for event in events:
                # Apply synaptic event, modifying the neuron's V, ge, gf, or gate
                event.affected_neuron.receive_synaptic_event(
                    event.synapse_type, event.weight
                )
                currently_affected_neurons.add(event.affected_neuron)

            # Collect all neurons that should be simulated
            neurons_to_simulate = currently_affected_neurons.union(active_state_neurons)
            # Prepare a set to hold the neurons turning active after this `dt`
            newly_active_state_neurons = set()

            for neuron in neurons_to_simulate:
                (V_after_update, spike) = neuron.update_and_spike(self.dt)
                # neuron.V is now V_after_update
                self._log_voltage_value(neuron=neuron, V=neuron.V, timestep=i)

                if spike:
                    self._log_spike_occurrence(neuron=neuron, t=t)
                    neuron.reset()  # V becomes Vreset, ge=0, gf=0, gate=0
                    for synapse in neuron.out_synapses:
                        self.event_queue.add_event(
                            time=t + synapse.delay,
                            neuron=synapse.post_neuron,
                            synapse_type=synapse.type,
                            weight=synapse.weight,
                        )

                # After update and potential reset, check if it remains internally active for the next step
                if neuron.ge != 0.0 or neuron.gf != 0.0 or neuron.gate != 0:
                    newly_active_state_neurons.add(neuron)

            active_state_neurons = newly_active_state_neurons

        if os.getenv("VIS", "0") == "1":
            self.launch_visualization()

    def simulatePlan(self, executionPlan: ExecutionPlan, dt=0.001) -> None:
        self.net = executionPlan.net
        self.dt = dt
        for trigger in executionPlan.input_triggers:
            self.apply_input_value(trigger.normalized_value, trigger.trigger_neuron)
        self.simulate(
            executionPlan.timeout,
        )

    def _log_spike_occurrence(self, neuron: ExplicitNeuron, t: float) -> None:
        if neuron.uid in self.spike_log:
            self.spike_log[neuron.uid].append(t)
        else:
            self.spike_log[neuron.uid] = [t]

    def _log_voltage_value(
        self, neuron: ExplicitNeuron, V: float, timestep: float
    ) -> None:
        self.voltage_log[neuron.uid].append((V, timestep))

    def launch_visualization(self):
        vis_topology(self.net)
        plot_chronogram(
            timesteps=self.timesteps,
            voltage_log=self.voltage_log,
            spike_log=self.spike_log,
        )
