from stick_emulator.primitives import DataEncoder
from stick_emulator.simulator import Simulator

from stick_emulator.compilation import Scalar, draw_comp_graph, compile_computation, report_spike_estimation, report_neuron_usage


if __name__ == "__main__":
    # 1. Computation
    x = Scalar(1.3)
    y = Scalar(2.4)

    out = x * y

    draw_comp_graph(out)

    # 2. Compile
    norm = 100
    execPlan = compile_computation(root=out, max_range=norm)

    report_spike_estimation(execPlan.net)
    report_neuron_usage(execPlan.net)


    # 3. Simulate
    # enc = DataEncoder()
    # sim = Simulator.init_with_plan(plan=execPlan, encoder=enc, dt=0.0005)
    # sim.simulate(simulation_time=600)

    # # 4. Readout
    # spikes_plus = sim.spike_log.get(execPlan.output_reader.read_neuron_plus.uid, [])
    # spikes_minus = sim.spike_log.get(execPlan.output_reader.read_neuron_minus.uid, [])

    # if len(spikes_plus) == 2:
    #     decoded_val = enc.decode_interval(spikes_plus[1] - spikes_plus[0])
    #     re_norm_value = decoded_val * 100
    #     print("Received plus output")
    #     print(f"{re_norm_value}")

    # if len(spikes_minus) == 2:
    #     decoded_val = enc.decode_interval(spikes_minus[1] - spikes_minus[0])
    #     re_norm_value = -1 * decoded_val * 100
    #     print("Received minus output")
    #     print(f"{re_norm_value}")



    
