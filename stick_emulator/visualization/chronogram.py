import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np


def plot_chronogram(
    timesteps: list[float],
    voltage_log: dict[str, list[float]],
    spike_log: dict[str, list[float]],
):
    n = len(voltage_log.keys())
    _, ax = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=(10, 5))
    colors = iter(cm.rainbow(np.linspace(0, 1, n)))

    for i, item in enumerate(voltage_log.keys()):
        c = next(colors)
        ax[i].plot(timesteps, voltage_log[item], c=c)

        # Neuron names
        ax[i].set_ylabel(item, rotation=0, labelpad=30)
        # Voltage limits
        ax[i].set_ylim(-15, 15)
        # Removing extra graphical elements (axes ticks and graph spines)
        ax[i].get_yaxis().set_ticks([])
        ax[i].xaxis.set_tick_params(length=0)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)

        if item in spike_log:
            for spike in spike_log[item]:
                ax[i].scatter(spike, 0, s=20, c=c)

    plt.show()
