from stick_emulator.primitives import DataEncoder
from stick_emulator.simulator import Simulator

from stick_emulator.compiler import (
    Scalar,
    draw_comp_graph,
    flatten,
    build_stick_net,
    get_output_reader,
    get_input_triggers,
    ExecutionPlan
)


if __name__ == "__main__":
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = Scalar(4.0)

    out = x - y

    # draw_comp_graph(out)

    norm = 100
    ops, conn, output_plug = flatten(out)

    net = build_stick_net(ops, conn, norm)
    input_triggers = get_input_triggers(ops, norm)
    output_reader = get_output_reader(output_plug, norm)

    if (not output_reader) or len(input_triggers) == 0:
        raise RuntimeError

    execPlan = ExecutionPlan(net, input_triggers, output_reader, timeout=600)

    enc = DataEncoder()
    sim = Simulator(net, enc)
    sim.simulatePlan(execPlan, dt=0.0005)

    spikes_plus = sim.spike_log.get(execPlan.output_reader.read_neuron_plus.uid, [])
    spikes_minus = sim.spike_log.get(execPlan.output_reader.read_neuron_minus.uid, [])

    if len(spikes_plus) == 2:
        interval = spikes_plus[1] - spikes_plus[0]
        print("Received plus output")
        decoded_val = enc.decode_interval(interval)
        re_norm_value = decoded_val * 100
        print(f"{re_norm_value}")

    if len(spikes_minus) == 2:
        interval = spikes_minus[1] - spikes_minus[0]
        print("Received minus output")
        decoded_val = enc.decode_interval(interval)
        re_norm_value = -1 * decoded_val * 100
        print(f"{re_norm_value}")
