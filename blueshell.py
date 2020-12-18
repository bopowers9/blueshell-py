"""A small spiking network implemented using numpy.

This module replicates the pool of spiking neurons described by Eugene Izhikevich in:

E. M. Izhikevich, "Simple model of spiking neurons," in IEEE Transactions on Neural Networks, vol. 14, no. 6, pp. 1569-1572, Nov. 2003, doi: 10.1109/TNN.2003.820440.

  Typical usage example:

  pool = IzhikevichPool(1000, 0.2, 2, 0.1)
  pool.update()
  ...
  pool.plot_spikes()
"""

import numpy as np
import math
import matplotlib.pyplot as plt


class IzhikevichPool:
    """A fixed number of neurons with synapses and external inputs."""

    def __init__(self, N, inhib_prop, inhib_strength_ratio, synaptic_density):
        """Instantiates the pool.

        Args:
            N (int): the total number of neurons.
            inhib_prop (float): the proportion of neurons that are inhibitory.
            inhib_strength_ratio (float): if > 1, inhibitory connections will
                be stronger. If < 1, inhibitory connections will be weaker.
            synaptic_density (float): Uniform r.v. from 0 (no connections) to
                1 (full connectivity).
        """

        Ni = int(math.floor(N * inhib_prop))
        Ne = N - Ni

        # Random values induce variation in neuron behavior
        re, ri = np.random.rand(Ne), np.random.rand(Ni)

        # Main model parameters
        # Here we use the defaults as described in the original paper
        self.a = np.append(0.02 * np.ones(Ne), 0.02 + 0.08 * ri)
        self.b = np.append(0.2 * np.ones(Ne), 0.25 - 0.05 * ri)
        self.c = np.append(-65 + 15 * re**2, -65 * np.ones(Ni))
        self.d = np.append(8 - 6 * re**2, 2 * np.ones(Ni))

        # Synapses are randomly initialized with specified density
        self.S = np.abs(np.random.randn(N, N))
        runif_mat = np.random.rand(N, N)
        zero_mask = np.nonzero(runif_mat > synaptic_density)
        self.S[zero_mask] = 0
        self.S[:, Ne:] *= -inhib_strength_ratio

        # External input
        self.external_input = np.zeros(N)

        # Store voltages in a numpy array
        self.history_limit = 1000
        self.voltage_history = np.zeros((N, self.history_limit))

        # Initial arrays for v and u
        self.v = -65 * np.ones(N)
        self.u = self.b * self.v

    def clamp_input(self, x):
        """Apply external inputs.

        Args:
            x (np.ndarray): input vector of the same length as self.v.
        """
        self.external_input = x

    def update(self, record_history=True):
        """Step the model forward by approximately 1 ms

        Args:
            record_history (bool): (optional) if true, save voltages.
        """
        # Detect and record spikes
        spiked = np.nonzero(self.v >= 30)

        # Save spike and voltage data, if applicable
        if record_history:
            self.voltage_history[:, :-1] = self.voltage_history[:, 1:]
            self.voltage_history[:, -1] = np.where(self.v < 30, self.v, 30)

        # Reset spiked neurons to their resting potential
        self.v[spiked] = self.c[spiked]
        self.u[spiked] += self.d[spiked]

        # Combine synaptic internal inputs with any external inputs
        internal_input = np.sum(self.S[:, spiked[0]], axis=1).flatten()
        I = internal_input + self.external_input

        # Update model, clipping huge values for numerical stability
        self.v = np.clip(self.v, -9999, 9999)

        # Finally, apply the magic formula to update v and u vectors
        self.v += (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I)
        self.u += (self.a * ((self.b * self.v) - self.u))

    def plot_spikes(self):
        """Use the eventplot() function in pyplot to visualize the raster plot.

        This plots whatever is in the 2D volgate_history array.
        """
        spike_events = [np.nonzero(x >= 30)[0] for x in self.voltage_history]
        plt.eventplot(spike_events, color='k', linewidths=2)
        plt.title('Spike Raster Plot')
        plt.xlabel('time (ms)')
        plt.ylabel('neuron index')
        plt.show()


if __name__ == "__main__":
    # Replicating the original paper...except synapses are Normally distributed
    # with more sparsity
    p = IzhikevichPool(1000, 0.2, 2, 0.1)

    # Run 1000 steps, each step representing 1 ms
    for i in range(1000):
        p.clamp_input(np.append(5.0 * np.random.randn(800), 2.0 * np.random.randn(200)))
        p.update()

    p.plot_spikes()
