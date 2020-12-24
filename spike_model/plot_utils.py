import numpy as np
import matplotlib.pyplot as plt



def plot_voltages(data):
    """Plot the voltages."""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.matshow(data, cmap='magma')
    ax.axis('off')
    plt.savefig('output.png')
    plt.close()



def plot_connectivity(weights, delays):
    """Plot weights and delays."""
    fig, ax = plt.subplots(1, 2)
    ax[0].matshow(weights)
    ax[0].set_title('weights')
    ax[1].matshow(delays)
    ax[1].set_title('delays')
    plt.show()
    plt.close()
