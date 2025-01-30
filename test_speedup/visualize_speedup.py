'''
This script visualizes the speedup achieved with Numba under varying parameters. The results can be seen in the 'plots' directory. 
'''

import numpy as np
import matplotlib.pyplot as plt


def read_txt_file(file_path):
    num_sims, time, batch_size, timesteps, no_hits_max, mold_cov_3d, mold_cov_surface = [
    ], [], [], [], [], [], []

    with open(file_path, 'r') as file:

        next(file)  # Skip first line

        for line in file:

            values = line.split()

            num_sims.append(int(values[0]))
            time.append(float(values[1]))
            batch_size.append(int(values[2]))
            timesteps.append(int(values[3]))
            no_hits_max.append(int(values[4]))
            mold_cov_3d.append(float(values[5]))
            mold_cov_surface.append(float(values[6]))

    # Convert lists to NumPy arrays
    num_sims = np.array(num_sims)
    time = np.array(time)
    batch_size = np.array(batch_size)
    timesteps = np.array(timesteps)
    no_hits_max = np.array(no_hits_max)
    mold_cov_3d = np.array(mold_cov_3d)
    mold_cov_surface = np.array(mold_cov_surface)
    return num_sims, time, batch_size, timesteps, no_hits_max, mold_cov_3d, mold_cov_surface


def plot_speedup_vs_numsims(time, time_njit, num_sims):
    speedup = time / time_njit
    avg_speedup_based_on_num_sims = []
    stddev_speedup_based_on_num_sims = []

    for n in np.unique(num_sims):
        avg_speedup_based_on_num_sims.append(np.mean(speedup[num_sims == n]))
        stddev_speedup_based_on_num_sims.append(np.std(speedup[num_sims == n]))

    num_sims_labels = [str(sim) for sim in np.unique(num_sims)]
    # plt.bar(num_sims_labels, avg_speedup_based_on_num_sims, color='skyblue', label='Speedup')
    plt.bar(num_sims_labels, avg_speedup_based_on_num_sims, yerr=stddev_speedup_based_on_num_sims,
            color='skyblue', capsize=5, label='Speedup')  # capsize adjusts the error bar caps

    plt.xlabel('Number of simulations')
    plt.ylabel('Speedup')
    plt.title(' Average speedup based on number of simulations')
    plt.legend()
    plt.show()


def plot_speedup_vs_numsims_batch_size(time, time_njit, num_sims, batch_size):
    speedup = time / time_njit
    avg_speedup_based_on_num_sims_100 = []
    stddev_speedup_based_on_num_sims_100 = []
    avg_speedup_based_on_num_sims_1000 = []
    stddev_speedup_based_on_num_sims_1000 = []

    unique_num_sims = np.unique(num_sims)
    for n in unique_num_sims:
        avg_speedup_based_on_num_sims_100.append(
            np.mean(speedup[(num_sims == n) & (batch_size == 100)])
        )
        stddev_speedup_based_on_num_sims_100.append(
            np.std(speedup[(num_sims == n) & (batch_size == 100)])
        )
        avg_speedup_based_on_num_sims_1000.append(
            np.mean(speedup[(num_sims == n) & (batch_size == 1000)])
        )
        stddev_speedup_based_on_num_sims_1000.append(
            np.std(speedup[(num_sims == n) & (batch_size == 1000)])
        )

    num_sims_labels = [str(sim) for sim in unique_num_sims]

    bar_width = 0.4
    x = np.arange(len(unique_num_sims))  # X positions for groups of bars

    plt.bar(x - bar_width / 2, avg_speedup_based_on_num_sims_100,
            yerr=stddev_speedup_based_on_num_sims_100,
            width=bar_width, color='skyblue', capsize=5, label='Batch size = 100')

    plt.bar(x + bar_width / 2, avg_speedup_based_on_num_sims_1000,
            yerr=stddev_speedup_based_on_num_sims_1000,
            width=bar_width, color='red', capsize=5, label='Batch size = 1000')

    plt.axhline(y=1, color='black', linestyle='--')  # Horizontal line at y=1
    plt.yscale('log')  # Log scale the vertical axis
    plt.xlabel('Number of simulations')
    plt.ylabel('Speedup')
    plt.title('Average speedup based on number of simulations and batch size')
    plt.xticks(x, num_sims_labels)  # Set x-axis labels to unique_num_sims
    plt.legend()
    plt.show()


def plot_speedup_vs_numsims_no_hits_max(time, time_njit, num_sims, no_hits_max):
    speedup = time / time_njit
    unique_num_sims = np.unique(num_sims)

    no_hits_max_values = np.unique(no_hits_max)
    avg_speedup_based_on_num_sims = np.zeros(
        (len(unique_num_sims), len(no_hits_max_values)))
    stddev_speedup_based_on_num_sims = np.zeros(
        (len(unique_num_sims), len(no_hits_max_values)))

    # Populate the arrays
    for i, n in enumerate(unique_num_sims):
        for j, h in enumerate(no_hits_max_values):
            avg_speedup_based_on_num_sims[i][j] = np.mean(
                speedup[(num_sims == n) & (no_hits_max == h)])
            stddev_speedup_based_on_num_sims[i][j] = np.std(
                speedup[(num_sims == n) & (no_hits_max == h)])

    # Plotting
    bar_width = 0.2  # Width of each bar
    x = np.arange(len(unique_num_sims))  # Positions for groups of bars

    # Create grouped bars
    for j, h in enumerate(no_hits_max_values):
        plt.bar(x + j * bar_width, avg_speedup_based_on_num_sims[:, j],
                yerr=stddev_speedup_based_on_num_sims[:, j],
                width=bar_width, capsize=5, label=f'No Hits Max = {h}')

    # Add labels, title, and legend
    plt.axhline(y=1, color='black', linestyle='--')  # Horizontal line at y=1
    plt.yscale('log')  # Log scale the vertical axis
    plt.xlabel('Number of simulations')
    plt.ylabel('Speedup')
    plt.title('Average speedup by num_sims and no_hits_max')
    plt.xticks(x + bar_width * (len(no_hits_max_values) - 1) / 2,
               [str(n) for n in unique_num_sims])  # Center x-axis labels
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    num_sims, time, batch_size, timesteps, \
        no_hits_max, mold_cov_3d, mold_cov_surface \
        = read_txt_file('results/result_normal.txt')

    num_sims_njit, time_njit, batch_size_njit, timesteps_njit, \
        no_hits_max_njit, mold_cov_3d_njit, mold_cov_surface_njit \
        = read_txt_file('results/result_njit.txt')

    plot_speedup_vs_numsims_batch_size(time, time_njit, num_sims, batch_size)
    plot_speedup_vs_numsims_no_hits_max(time, time_njit, num_sims, no_hits_max)
