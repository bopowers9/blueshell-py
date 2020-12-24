import numpy as np
import math
import matplotlib.pyplot as plt


def shuffle_matrix(matrix):
    """Simply shuffles item order while preserving network topology."""
    new_matrix = np.zeros(matrix.shape)
    n = matrix.shape[0]
    # Get new indices from old by accessing order map: new = order_map[old]
    order_map = np.arange(n)
    np.random.shuffle(order_map)
    for i in np.arange(n):
        for j in np.arange(n):
            new_i = order_map[i]
            new_j = order_map[j]
            new_matrix[new_i, new_j] = matrix[i, j]

    return new_matrix


def build_small_world_graph(config):
    """Implements Watts-Strogatz algorithm to construct a small world graph."""
    Ne = config["N"]["e"]
    Ni = config["N"]["i"]
    n = Ne + Ni
    k = config["small-world-params"]["k"]
    p = config["small-world-params"]["p"]
    # Initialize dense adjacency matrix
    adj_matrix = np.zeros([n, n])
    # Connect each neuron to k nearest neighbors in ring lattice
    halfk = int(math.floor(k / 2))
    ring = np.roll(np.arange(n), halfk)
    for i in range(n):
        neighbors = np.append(ring[:halfk], ring[halfk + 1:2 * halfk + 1])
        adj_matrix[i, neighbors] = 1
        ring = np.roll(ring, -1)

    # Scan through edges, rewiring them with probability p
    edge_indices = np.nonzero(adj_matrix)
    for i, j in zip(list(edge_indices[0]), list(edge_indices[1])):
        if np.random.rand() < p:
            # Rewire
            adj_matrix[i, j] = 0
            # Keep i but resample j
            candidates = np.arange(n)
            # Ensure that candidates does not include i
            candidates = np.append(candidates[:i], candidates[i + 1:])
            new_j = np.random.choice(candidates, 1)
            adj_matrix[i, new_j] = 1

    return adj_matrix
