from stick_emulator.primitives import ExplicitNeuron, SpikingNetworkModule


class NeuronHeader:
    def __init__(self, neuron_plus: ExplicitNeuron, neuron_minus: ExplicitNeuron):
        self.neuron_plus = neuron_plus
        self.neuron_minus = neuron_minus


class InputTrigger:
    def __init__(self, value: float, norm: float, neuron_header: NeuronHeader):
        assert value / norm <= 1.0, f"Input value outside range [0, {norm}]"
        assert (
            norm <= 100
        ), f"Guardrail: normalization only tested up to 100; {norm} given"

        self.normalized_value = abs(value) / norm

        if value >= 0:
            self.trigger_neuron = neuron_header.neuron_plus
        else:
            self.trigger_neuron = neuron_header.neuron_minus


class OutputReader:
    def __init__(self, header: NeuronHeader, norm: float):
        self.read_neuron_plus = header.neuron_plus
        self.read_neuron_minus = header.neuron_minus

        self.normalization = norm


class ExecutionPlan:
    def __init__(
        self,
        net: SpikingNetworkModule,
        triggers: list[InputTrigger],
        reader: OutputReader,
        timeout: float = 400,
    ):
        self.net = net
        self.input_triggers = triggers
        self.output_reader = reader
        self.timeout = timeout
