from spike_model import pool
import numpy as np

if __name__ == "__main__":
    # Load pool configuration from file
    config = pool.load_pool_config('pool-config.yaml')

    # Create pool instance
    pool = pool.IzhikevichPool(config)

    # pool.plot_connectivity()

    # Simulate
    duration_ms = 1000
    for i in range(duration_ms):
        # Simulate some random thalamic input
        pool.clamp_input(np.append(
            5 * np.random.rand(config["N"]["e"]),
            5 * np.random.rand(config["N"]["i"]),
        ))
        # Step model forward
        pool.update()

    # Plot voltages
    pool.plot_voltages()
