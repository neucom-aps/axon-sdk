# Axon SDK — A simulation framework for spike-timing-based neural computation using the STICK model.
# Copyright (C) 2024–2025 Neucom ApS
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from axon_sdk.primitives import DataEncoder
from axon_sdk.simulator import Simulator

from axon_sdk.compilation import Scalar, compile_computation


if __name__ == "__main__":
    # 1. Computation
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = Scalar(4.0)

    out = (x + y) * z

    out.draw_comp_graph(outfile='basic_computation_graph')

    # 2. Compile
    norm = 100
    execPlan = compile_computation(root=out, max_range=norm)

    # 3. Simulate
    enc = DataEncoder()
    sim = Simulator.init_with_plan(execPlan, enc)
    sim.simulate(simulation_time=600)

    # 4. Readout
    spikes_plus = sim.spike_log.get(execPlan.output_reader.read_neuron_plus.uid, [])
    spikes_minus = sim.spike_log.get(execPlan.output_reader.read_neuron_minus.uid, [])

    if len(spikes_plus) == 2:
        decoded_val = enc.decode_interval(spikes_plus[1] - spikes_plus[0])
        re_norm_value = decoded_val * 100
        print("Received plus output")
        print(f"{re_norm_value}")

    if len(spikes_minus) == 2:
        decoded_val = enc.decode_interval(spikes_minus[1] - spikes_minus[0])
        re_norm_value = -1 * decoded_val * 100
        print("Received minus output")
        print(f"{re_norm_value}")
