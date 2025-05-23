from stick_emulator.networks import LinearCombinatorNetwork
from stick_emulator.primitives import DataEncoder

from typing import Optional


class AdderNetwork(LinearCombinatorNetwork):
    """
    Wrapper around the linear combinator that performs signed 2-addition:
    out = x1 + x2
    """
    def __init__(self, encoder: DataEncoder, module_name: Optional[str] = None):
        coeffs = [1.0, 1.0]
        super().__init__(encoder=encoder, N=2, coeff=coeffs, module_name=module_name)

        self.inpu1_plus = self.input_plus[0]
        self.inpu1_minus = self.input_minus[0]
        self.inpu2_plus = self.input_plus[1]
        self.inpu2_minus = self.input_minus[1]


if __name__ == "__main__":
    from stick_emulator.simulator import Simulator
    inp1 = -0.0001
    inp2 = 0.4

    assert inp1 + inp2 <= 1.0, "inp1 + inp2 must be smaller than 1.0"

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = AdderNetwork(encoder, module_name='adder_net')
    sim = Simulator(net, encoder, dt=0.01)

    if inp1 >= 0:
        sim.apply_input_value(abs(inp1), net.inpu1_plus)
    else:
        sim.apply_input_value(abs(inp1), net.inpu1_minus)

    if inp2 >= 0:
        sim.apply_input_value(abs(inp2), net.inpu2_plus)
    else:
        sim.apply_input_value(abs(inp2), net.inpu2_minus)

    sim.simulate(450)

    print("\n==========================")
    expected = inp1 + inp2
    print(f"Expected linear combination: {expected:.3f}")

    plus_spikes = sim.spike_log.get(net.output_plus.uid, [])
    minus_spikes = sim.spike_log.get(net.output_minus.uid, [])

    if expected > 0:
        assert len(plus_spikes) == 2, "Expected 2 + spikes since expected is > 0"
        interval = plus_spikes[1] - plus_spikes[0]
        print(f"Decoded value: {encoder.decode_interval(interval)}")
    else:
        assert len(minus_spikes) == 2, "Expected 2 - spikes since expected is > 0"
        interval = minus_spikes[1] - minus_spikes[0]
        print(f"Decoded value: {encoder.decode_interval(interval)}")
