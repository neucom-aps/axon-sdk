import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def build_array(length: int, entry_points: list[tuple[float, float]]) -> list[float]:
    """
    Example:
    entry_points = [(4.0, 1), (0.0, 3), (5.0, 6)]
    length = 10
    -> output = [0.0, 4.0, 4.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0]
    """
    entry_points_sorted = sorted(entry_points, key=lambda pair: pair[1])
    entry_points_dict = {t[1]: t[0] for t in entry_points_sorted}

    result = [0.0] * length
    last_valid_value = 0.0
    for i in range(length):
        if (val := entry_points_dict.get(i, None)) is not None:
            last_valid_value = val

        result[i] = last_valid_value

    return result


def plot_chronogram(
    timesteps: list[float],
    voltage_log: dict[str, list[tuple]],
    spike_log: dict[str, list[float]],
):
    print("Launching chronogram visualization...")
    print("=========================================")
    n = len(voltage_log.keys())
    _, ax = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=(10, 5))
    values = [i / (n - 1) for i in range(n)]  # linearly spaced values between 0 and 1
    colors = iter(cm.rainbow(values))

    for i, item in enumerate(voltage_log.keys()):
        c = next(colors)
        v_log = voltage_log[item]
        if len(timesteps) != len(v_log):
            v_log = build_array(len(timesteps), v_log)

        # Neuron names
        ax[i].set_ylabel(item, rotation=0, labelpad=30)
        # Voltage limits
        ax[i].set_ylim(-12, 12)
        ax[i].axhline(0, color="lightgray", linestyle="--")
        # Removing extra graphical elements (axes ticks and graph spines)
        ax[i].get_yaxis().set_ticks([])
        ax[i].xaxis.set_tick_params(length=0)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)

        ax[i].plot(timesteps, v_log, c="#2A868C")

        if item in spike_log:
            for spike in spike_log[item]:
                ax[i].scatter(spike, 0, s=30, c="#2A868C")

    plt.legend()
    plt.show()
    print("=========================================")
