from axon_sdk.compilation import Scalar, compile_computation
from axon_sdk.simulator import Simulator, decode_output, count_spikes
from axon_sdk.primitives import DataEncoder

from axon_sdk.usagereport import (
    report_neuron_usage,
    report_spike_estimation_for_net,
    report_energy_and_latency_estimation_for_net,
)

import os


def regular_matmul(A: list[list[Scalar]], B: list[list[Scalar]]) -> list[list[Scalar]]:
    C = [[Scalar(0.0), Scalar(0.0)], [Scalar(0.0), Scalar(0.0)]]

    C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0]
    C[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1]
    C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0]
    C[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1]

    return C


def strassen_matmul(A: list[list[Scalar]], B: list[list[Scalar]]) -> list[list[Scalar]]:
    M1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1])
    M2 = (A[1][0] + A[1][1]) * B[0][0]
    M3 = A[0][0] * (B[0][1] - B[1][1])
    M4 = A[1][1] * (B[1][0] - B[0][0])
    M5 = (A[0][0] + A[0][1]) * B[1][1]
    M6 = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1])
    M7 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1])

    c11 = M1 + M4 - M5 + M7
    c12 = M3 + M5
    c21 = M2 + M4
    c22 = M1 - M2 + M3 + M6

    C = [[c11, c12], [c21, c22]]

    return C


def print_mat(M: list[list[Scalar]]) -> None:
    for i in range(len(M)):
        for j in range(len(M[0])):
            print(M[i][j].data, end=" ")
        print()
    print()


def sum_mat(M: list[list[Scalar]]) -> Scalar:
    out = Scalar(0)
    for i in range(len(M)):
        for j in range(len(M[0])):
            out += M[i][j]

    return out


if __name__ == "__main__":
    a11 = Scalar(2.0)
    a12 = Scalar(3.0)
    a21 = Scalar(2.0)
    a22 = Scalar(1.0)

    b11 = Scalar(1.0)
    b12 = Scalar(2.0)
    b21 = Scalar(3.0)
    b22 = Scalar(2.0)

    A = [[a11, a12], [a21, a22]]
    B = [[b11, b12], [b21, b22]]

    print_mat(A)
    print_mat(B)

    C = regular_matmul(A, B)
    C_str = strassen_matmul(A, B)

    print_mat(C)
    print_mat(C_str)

    out = sum_mat(C_str)

    # out.draw_comp_graph(outfile='matmul_2x2')

    plan = compile_computation(out, max_range=100)
    enc = DataEncoder()
    sim = Simulator.init_with_plan(plan=plan, encoder=enc)

    report_neuron_usage(plan.net)
    report_spike_estimation_for_net(plan.net)
    report_energy_and_latency_estimation_for_net(plan.net)

    # os.environ["VIS"] = "1"
    sim.simulate(simulation_time=3000)

    output = decode_output(sim=sim, reader=plan.output_reader)

    num_spikes = count_spikes(sim)
    print(f"Actual number of processed spikes: {num_spikes}")
    
