import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stick_emulator.compilation import Scalar, draw_comp_graph
from stick_emulator.compilation import report_neuron_usage, report_spike_estimation, compile_computation, ExecutionPlan


def mul_mat(a, b):
    a_shape1 = len(a)
    a_shape2 = len(a[0]) if isinstance(a[0], list) else None
    b_shape1 = len(b)
    b_shape2 = len(b[0]) if isinstance(b[0], list) else 0

    assert (
        a_shape2 == b_shape1
    ), f"Error: a: {a_shape1}x{a_shape2}, b: {b_shape1}x{b_shape2}"

    c = [[0 for _ in range(b_shape2)] for _ in range(a_shape1)]

    for i in range(a_shape1):
        for j in range(b_shape2):
            for k in range(a_shape2):
                c[i][j] += a[i][k] * b[k][j]

    return c


def div_scalar(mat_a, scalar):
    shape1 = len(mat_a)
    shape2 = len(mat_a[0])

    out = [[0 for _ in range(shape2)] for _ in range(shape1)]

    for i in range(shape1):
        for j in range(shape2):
            out[i][j] = mat_a[i][j] / scalar

    return out


def mul_scalar(mat_a, scalar):
    shape1 = len(mat_a)
    shape2 = len(mat_a[0])

    out = [[0 for _ in range(shape2)] for _ in range(shape1)]

    for i in range(shape1):
        for j in range(shape2):
            out[i][j] = mat_a[i][j] * scalar

    return out


def trp_mat(a):
    a_shape1 = len(a)
    a_shape2 = len(a[0]) if isinstance(a[0], list) else None

    c = [[0 for _ in range(a_shape1)] for _ in range(a_shape2)]

    for i in range(a_shape1):
        for j in range(a_shape2):
            c[j][i] = a[i][j]

    return c


def sum_mat(a, b):
    a_shape1 = len(a)
    a_shape2 = len(a[0]) if isinstance(a[0], list) else None
    b_shape1 = len(b)
    b_shape2 = len(b[0]) if isinstance(b[0], list) else 0

    assert (
        a_shape1 == b_shape1
    ), f"Error: a: {a_shape1}x{a_shape2}, b: {b_shape1}x{b_shape2}"
    assert (
        a_shape2 == b_shape2
    ), f"Error: a: {a_shape1}x{a_shape2}, b: {b_shape1}x{b_shape2}"

    c = [[0 for _ in range(b_shape2)] for _ in range(a_shape1)]

    for i in range(a_shape1):
        for j in range(b_shape2):
            c[i][j] = a[i][j] + b[i][j]

    return c


def predict(x, P, F, Q):
    assert len(x) == 2 and len(x[0]) == 1
    assert len(F) == 2 and len(F[0]) == 2
    # x = F * x
    x = mul_mat(F, x)

    P = sum_mat(mul_mat(mul_mat(F, P), trp_mat(F)), Q)

    return x, P


def update(x, P, z, R, H):
    assert len(z) == 1
    assert isinstance(R, float)

    y = [z[0] - mul_mat(H, x)[0][0]]
    S = [mul_mat(mul_mat(H, P), trp_mat(H))[0][0] + R]
    K = div_scalar(mul_mat(P, trp_mat(H)), S[0])

    x_out = sum_mat(x, mul_scalar(K, y[0]))

    KH = mul_mat(K, H)

    eye = [[1, 0], [0, 1]]
    I_KH = sum_mat(eye, mul_scalar(KH, -1))

    T1 = mul_mat(mul_mat(I_KH, P), trp_mat(I_KH))
    T2 = mul_mat(mul_scalar(K, R), trp_mat(K))
    P_out = sum_mat(T1, T2)

    return x_out, P_out


def kalman_step(
    noisy_item: Scalar,
    x: list[list[Scalar]],
    P: list[list[Scalar]],
):
    lambda_sq = 0.001
    sigma_sq = 20.0

    # State transition matrix
    F = [
        [2.0, -1.0],
        [1.0, 0.0],
    ]

    # Measurement function
    H = [[1.0, 0.0]]

    # Process noise
    Q = [
        [lambda_sq, 0.0],
        [0.0, 0.0],
    ]

    # Measurement noise
    R = sigma_sq

    x, P = predict(x, P, F, Q)
    z = [noisy_item]
    x, P = update(x, P, z, R, H)

    return x, P


if __name__ == "__main__":
    # Initial condition
    x = [[Scalar(11.34)], [Scalar(12.64)]]  # arbitrary values
    # Covariance matrix
    P = [[Scalar(1.0), Scalar(0.0)], [Scalar(1.0), Scalar(1.0)]]

    noisy_item = Scalar(13.54)
    new_x, new_P = kalman_step(noisy_item, x, P)

    # auxiliary var to display all output in the same graph
    out = new_x[0][0] + new_P[0][0] + new_P[0][1] + new_P[1][0] + new_P[1][1]

    draw_comp_graph(out, outfile='kalman', rankdir='TB')

    plan = compile_computation(out, max_range=100)

    report_neuron_usage(plan.net)
    report_spike_estimation(plan.net)


