import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np


class Chronogram:
    def __init__(
        self,
        timesteps: list[float],
        spike_log: dict[str, list[float]],
        voltage_log: dict[str, list[float]],
        Vt: float,
        dt: float,
    ):
        self.timesteps = timesteps  # [0 :: int(1 / dt)]
        self.spike_log = spike_log
        self.voltage_log = voltage_log
        self.Vt = Vt

    def show(self):
        n = len(self.voltage_log.keys())
        fig, ax = plt.subplots(
            nrows=n,
            ncols=1,
            sharex=True,
        )
        colors = iter(cm.rainbow(np.linspace(0, 1, n)))

        for i, item in enumerate(self.voltage_log.keys()):
            c = next(colors)
            ax[i].plot(self.timesteps, self.voltage_log[item], c=c)

            # Neuron names
            ax[i].set_ylabel(item.split("_")[-2], rotation=0, labelpad=10)
            # Voltage limits
            ax[i].set_ylim(-15, 15)
            # Removing extra graphical elements (axes ticks and graph spines)
            ax[i].get_yaxis().set_ticks([])
            ax[i].xaxis.set_tick_params(length=0)
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["right"].set_visible(False)
            ax[i].spines["bottom"].set_visible(False)
            ax[i].spines["left"].set_visible(False)

            if item in self.spike_log:
                for spike in self.spike_log[item]:
                    ax[i].scatter(spike, 0, s=20, c=c)

        plt.show()
