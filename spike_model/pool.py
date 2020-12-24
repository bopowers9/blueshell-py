import numpy as np
import matplotlib.pyplot as plt
import sys
import yaml
from . import graph_utils
from . import plot_utils

def load_pool_config(file_path):
    """Load parameters dict from the specified yaml file."""
    config = None
    with open(file_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    if config:
        return config
    else:
        sys.exit(0)


def create_neuron_arrays(config):
    """Initializes a, b, c, d, v and u arrays for the Izhikevich model."""
    Ne = config["N"]["e"]
    Ni = config["N"]["i"]

    a = np.append(
        config["a"]["e"] * np.ones(Ne),
        config["a"]["i"] * np.ones(Ni),
    )

    b = np.append(
        config["b"]["e"] * np.ones(Ne),
        config["b"]["i"] * np.ones(Ni),
    )

    c = np.append(
        config["c"]["e"] * np.ones(Ne),
        config["c"]["i"] * np.ones(Ni),
    )

    d = np.append(
        config["d"]["e"] * np.ones(Ne),
        config["d"]["i"] * np.ones(Ni),
    )

    v = np.append(
        config["v-init"]["e"] * np.ones(Ne),
        config["v-init"]["i"] * np.ones(Ni),
    )

    u = b * v

    return a, b, c, d, v, u


def create_weights(config):
    """Initializes an adjacency matrix."""
    Ne = config["N"]["e"]
    Ni = config["N"]["i"]
    e_scale = config["scale"]["e"]
    i_scale = config["scale"]["i"]
    architecture = config["architecture"]

    if architecture == "small-world":
        # Create binary weights with a small world structure
        adj_matrix = graph_utils.build_small_world_graph(config)
        # Shuffle neuron order to mix up E and I neurons within local clusters
        # while taking care to preserve the global small world topology
        adj_matrix = graph_utils.shuffle_matrix(adj_matrix)

    # Make inhibitory weights negative
    adj_matrix[:, Ne:] *= -1
    # Sample continuous magnitudes from uniform distribution and scale weights
    adj_matrix[:, :] *= np.random.rand(Ne + Ni, Ne + Ni)
    adj_matrix[:, :Ne] *= e_scale
    adj_matrix[:, Ne:] *= i_scale

    return adj_matrix


def create_delays(adj_matrix, config):
    """Create a weight matrix to store delay in ms for each synapse."""
    Ne = config["N"]["e"]
    Ni = config["N"]["i"]
    e_min = config["delay"]["e"]["min"]
    e_max = config["delay"]["e"]["max"]
    i_min = config["delay"]["i"]["min"]
    i_max = config["delay"]["i"]["min"]

    delays = np.where(adj_matrix != 0, 1, 0)
    delays[:, :Ne] *= np.random.randint(e_min, e_max + 1, size=(Ne + Ni, Ne))
    delays[:, Ne:] *= np.random.randint(i_min, i_max + 1, size=(Ne + Ni, Ni))

    return delays


def create_horizon(config):
    """The horizon accumulates inputs for each neuron over the near future."""
    Ne = config["N"]["e"]
    Ni = config["N"]["i"]
    e_min = config["delay"]["e"]["min"]
    e_max = config["delay"]["e"]["max"]
    i_min = config["delay"]["i"]["min"]
    i_max = config["delay"]["i"]["min"]

    max_delay_steps = max(e_max, i_max)
    horizon = np.zeros((max_delay_steps + 1, Ne + Ni))

    return horizon


def collect_future_inputs(fired, weights, delays, horizon):
    """Add outgoing weight values for all firings to the horizon given delays."""
    # Iterate through all fired neurons by index
    for pre_index in fired[0]:
        # Get weights and delays for synapses emanating from this fired neuron
        post_indices = np.nonzero(weights[:, pre_index])[0]
        values = weights[post_indices, pre_index].flatten()
        delay_amounts = delays[post_indices, pre_index].flatten()
        # Assign to horizon matrix
        horizon[delay_amounts, post_indices] += values

    return horizon


def advance_horizon(horizon):
    """Shift all horizon rows downward. Performed once each step."""
    horizon[:-1, :] = horizon[1:, :]
    horizon[-1, :] = np.zeros(horizon.shape[1])

    return horizon


class IzhikevichPool:
    def __init__(self, config):
        self.Ne, self.Ni = config["N"]["e"], config["N"]["i"]
        # Initialize arrays for the spiking model
        a, b, c, d, v, u = create_neuron_arrays(config)
        self.a, self.b, self.c, self.d = a, b, c, d
        self.v, self.u = v, u
        # Initialize synaptic weights
        self.weights = create_weights(config)
        # Assign delays to synapses
        self.delays = create_delays(self.weights, config)
        # Initialize horizon matrix to aggregate delayed input values
        self.horizon = create_horizon(config)
        # External input
        self.ext_input = np.zeros(self.v.size)
        # Voltage history data
        self.history = np.zeros(
            [self.Ne + self.Ni, int(config["history-limit"])]
        )
        # Step counter
        self.t = 0

    def clamp_input(self, x):
        self.ext_input = x

    def plot_connectivity(self):
        plot_utils.plot_connectivity(self.weights, self.delays)

    def plot_voltages(self):
        plot_utils.plot_voltages(self.history)

    def update(self, record_history=True):
        # Identify neurons that have just fired action potentials
        fired = np.nonzero(self.v >= 30)
        # Reset fired neurons to base potential
        self.v[fired] = self.c[fired]
        self.u[fired] += self.d[fired]

        # Update the horizon with future inputs
        self.horizon = collect_future_inputs(
            fired,
            self.weights,
            self.delays,
            self.horizon,
        )

        # Merge external and synaptic inputs
        # Synaptic inputs at this moment are stored in zeroth row of horizon
        I = self.ext_input + self.horizon[0, :]

        # Clip large values
        self.v = np.clip(self.v, -999, 999)

        # Update u and v
        self.v += (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I)
        self.u += (self.a * ((self.b * self.v) - self.u))

        # Roll the horizon forward in preparation for the next step
        self.horizon = advance_horizon(self.horizon)

        # Tick forward
        self.t += 1

        if record_history:
            # Shift existing raster data to the left
            self.history[:, :-1] = self.history[:, 1:]
            # Store current voltages in the last column
            self.history[:, -1] = np.where(self.v < 30, self.v, 30)
