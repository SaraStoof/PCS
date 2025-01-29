'''
The main file for the simulation, uses the other helper functions to run the simulations
'''

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import time
import sys
from scipy.stats import sem 
from helpers.helpers_single_value import *
from helpers.helpers_plots import *
from helpers.helpers_loop import *
from helpers.helpers_user_input import *


TIMESTEPS_PER_DAY = 20

GRID_X, GRID_Y, GRID_Z = 100, 100, 100  # The max index of each axis

# To spawn in a corner
# SPAWN_X, SPAWN_Y, SPAWN_Z = 0, 0, 0
# SPAWN_ON_X_EDGE, SPAWN_ON_Y_EDGE, SPAWN_ON_Z_EDGE = (True, False), (True, False), (True, False)

# to spawn at an edge
# SPAWN_X, SPAWN_Y, SPAWN_Z = 0, 0, 50
# SPAWN_ON_X_EDGE, SPAWN_ON_Y_EDGE, SPAWN_ON_Z_EDGE = (True, False), (True, False), (False, False)

# To spawn at a surface
SPAWN_X, SPAWN_Y, SPAWN_Z = 50, 50, 100
SPAWN_ON_X_EDGE, SPAWN_ON_Y_EDGE, SPAWN_ON_Z_EDGE = (False, False), (False, False), (False, True)


MAX_RADIUS = (min(GRID_X, GRID_Y, GRID_Z) // 2) + 5

NUM_SIMS = 5
TEMP = 30
RH = 97
BATCH_SIZE = 1000
NO_HITS_MAX = 5
DAYS = 6
TIMESTEPS = DAYS * TIMESTEPS_PER_DAY


@njit(parallel=True)
def decay_grid(grid):
    '''
    Introduces decay to the grid, setting certain particles to die by prioritizing
    particles around the edge of the cluster
    '''
    decay_amount = 0
    sum_grid = int(np.sum(grid))
    for _ in prange(sum_grid):
        if np.random.uniform() < DECAY_PROB:
            decay_amount += 1
    if decay_amount == 0:
        return
    # find middle point
    sum_grid = int(np.sum(grid))
    coords = np.zeros((sum_grid, 3))
    idx = 0
    x_avg = 0
    y_avg = 0
    z_avg = 0
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                if grid[x, y, z] == 1:
                    coords[idx] = np.array((x, y, z), dtype=np.int32)
                    x_avg += x
                    y_avg += y
                    z_avg += z
                    idx += 1

    x_avg /= idx
    y_avg /= idx
    z_avg /= idx
    middle = (x_avg, y_avg, z_avg)

    # keep removing furthest point from middle point
    distances = np.zeros(coords.shape[0])
    for i in prange(coords.shape[0]):
        distances[i] = np.sqrt((coords[i][0] - middle[0]) ** 2 +
                               (coords[i][1] - middle[1]) ** 2 +
                               (coords[i][2] - middle[2]) ** 2)
    for _ in range(decay_amount):
        idx = np.argmax(distances)
        furthest = coords[idx]
        grid[int(furthest[0]), int(furthest[1]), int(furthest[2])] = 0
        distances[idx] = -1


# This decorator tells Numba to compile this function using the JIT (just-in-time) compiler
@njit
def particle_loop(grid, ATTACH_PROB, DECAY_PROB, batch_size=1000):
    '''
    This is the main loop of the simulation, it runs the simulation for mold growth for
    a certain amount of timesteps by making changes to the inputted grid
    '''
    reached_edge = False
    # spawns particles closer to where the seed is, to speed up the program.
    current_radius = 5
    particle_count = 0
    m_history_3d = np.zeros(DAYS)
    m_history_surf = np.zeros(DAYS)

    for i in prange(TIMESTEPS):
        if i % TIMESTEPS_PER_DAY == 0:
            #These things happen once a day
            decay_grid(grid)
            m_history_3d[i // TIMESTEPS_PER_DAY] = coverage_to_m_value(mold_coverage(grid))
            m_history_surf[i // TIMESTEPS_PER_DAY] = coverage_to_m_value(mold_cov_surface(grid[:, :, GRID_Z]) + mold_cov_surface(grid[:, :, 0]) + mold_cov_surface(grid[GRID_Z, :, :])
                          + mold_cov_surface(grid[0, :, :]) + mold_cov_surface(grid[:, GRID_Y, :]) + mold_cov_surface(grid[:, 0, :]))
        # http://datagenetics.com/blog/january32020/index.html

        # Theta is the angle in the x-y plane
        # Phi is the angle from the z-axis
        theta = np.random.uniform(0, np.pi * 2, batch_size)
        phi = np.random.uniform(np.pi, 2 * np.pi, batch_size)

        # Initialize an empty array to hold the particle coordinates
        particles = np.zeros((batch_size, 3))

        if reached_edge == False:
            # Populate the particle array manually
            particles[:, 0] = new_x_coords(theta, phi, current_radius,
                                           SPAWN_ON_X_EDGE, SPAWN_X)
            particles[:, 1] = new_y_coords(theta, phi, current_radius,
                                           SPAWN_ON_Y_EDGE, SPAWN_Y)
            particles[:, 2] = new_z_coords(phi, current_radius,
                                           SPAWN_ON_Z_EDGE, SPAWN_Z)

        else:
            particles[:, 0] = (np.random.randint(0, GRID_X, batch_size))
            particles[:, 1] = (np.random.randint(0, GRID_Y, batch_size))
            particles[:, 2] = (np.random.randint(0, GRID_Z, batch_size))

        if len(particles) > current_radius**3:
            particles = particles[:current_radius**3]

        particles = np.floor(particles)
        particle_count += len(particles)

        particles = in_bounds(particles, current_radius, SPAWN_X, SPAWN_Y, SPAWN_Z,
                              SPAWN_ON_X_EDGE, SPAWN_ON_Y_EDGE, SPAWN_ON_Z_EDGE)

        no_hits_count = 0

        while len(particles) > 0:

            particles = move(particles)

            particles = in_bounds(particles, current_radius, SPAWN_X, SPAWN_Y, SPAWN_Z,
                                  SPAWN_ON_X_EDGE, SPAWN_ON_Y_EDGE, SPAWN_ON_Z_EDGE)

            # check neighbors and update grid
            hits, p_indices = check_neighbor(particles, grid,
                                             GRID_X, GRID_Y, GRID_Z, ATTACH_PROB)

            # Break if particles have moved five turns with no hits.
            if len(hits) == 0:
                no_hits_count += 1
                if no_hits_count > NO_HITS_MAX:
                    break
            else:
                no_hits_count = 0

            # Update grid
            for hit in hits:
                x, y, z = int(hit[0]), int(hit[1]), int(hit[2])
                grid[x, y, z] = 1
                dist_to_seed = np.linalg.norm(hit - np.array([SPAWN_X, SPAWN_Y, SPAWN_Z]))
                if dist_to_seed >= current_radius - 1 and reached_edge == False:
                    current_radius += 5
                    if current_radius >= MAX_RADIUS:
                        reached_edge = True

            # Remove particles that already attached themselves to the cluster
            particles = remove_indices(particles, p_indices)
    return m_history_3d, m_history_surf


@njit(parallel=True)
def monte_carlo(ATTACH_PROB, DECAY_PROB):
    '''
    This function runs the simulation a certain amount of times and returns the average
    grid and mold coverage
    '''
    aggr_grid = np.zeros((GRID_X + 1, GRID_Y + 1, GRID_Z + 1))
    mold_cov_3d = 0
    mold_cov_surf = 0
    history_3d = np.zeros((NUM_SIMS, DAYS))
    history_surf = np.zeros((NUM_SIMS, DAYS))

    for i in prange(NUM_SIMS):
        # Initialize grid (plus 1 to account for 0-index)
        grid = np.zeros((GRID_X + 1, GRID_Y + 1, GRID_Z + 1))
        grid[SPAWN_X, SPAWN_Y, SPAWN_Z] = 1
        history_3d[i], history_surf[i] = particle_loop(grid, ATTACH_PROB, DECAY_PROB)

        aggr_grid += grid
        mold_cov_3d += mold_coverage(grid)
        
        mold_cov_surf += (mold_cov_surface(grid[:, :, GRID_Z]) + mold_cov_surface(grid[:, :, 0]) + mold_cov_surface(grid[GRID_Z, :, :])
                          + mold_cov_surface(grid[0, :, :]) + mold_cov_surface(grid[:, GRID_Y, :]) + mold_cov_surface(grid[:, 0, :]))

    aggr_grid = aggr_grid/NUM_SIMS
    mold_cov_3d = mold_cov_3d / NUM_SIMS
    mold_cov_surf = mold_cov_surf/ NUM_SIMS
    return aggr_grid, mold_cov_3d, mold_cov_surf, history_3d, history_surf


def test_3d():
    global NUM_SIMS, BATCH_SIZE, TEMP, RH, DAYS, TIMESTEPS, TIMESTEPS_PER_DAY
    global NO_HITS_MAX, ATTACH_PROB, DECAY_PROB

    temp_list = [5, 30]
    rh_list = [80, 90, 97, 100]
    DAYS = 168
    TIMESTEPS_PER_DAY = 5
    TIMESTEPS = DAYS * TIMESTEPS_PER_DAY
    NUM_SIMS = 100
    BATCH_SIZE = 1000

    titles = ['Mould Coverage of 3D simulation', 'Surface Mold Coverage of 3D Simulation']
    for i in range(len(temp_list)):
        temp = temp_list[i]
        TEMP = temp
        fig, axes = plt.subplots(1, 2, figsize=(12, 5)) 
        for rh in rh_list:
            RH = rh
            ATTACH_PROB = get_attach_prob(TEMP, RH)
            DECAY_PROB = get_decay_prob(ATTACH_PROB, 0.05, 10)
            _, _, _, history_3d, history_surf = monte_carlo(ATTACH_PROB, DECAY_PROB)
            m_mean_3d = np.mean(history_3d, axis=0)
            m_mean_surf = np.mean(history_surf, axis=0)
            ci_3d = 1.96 * sem(history_3d, axis=0)
            ci_surf = 1.96 * sem(history_surf, axis=0)
            # Plot mean curve
            axes[0].plot(m_mean_3d, linestyle='-', label=f"RH = {rh}%, Temp = {temp}°C")
            axes[0].fill_between(range(len(m_mean_3d)), m_mean_3d - ci_3d, m_mean_3d + ci_3d, alpha=0.2)
            
            # Plotting mean and confidence intervals for Surface
            axes[1].plot(m_mean_surf, linestyle='-', label=f"RH = {rh}%, Temp = {temp}°C")
            axes[1].fill_between(range(len(m_mean_surf)), m_mean_surf - ci_surf, m_mean_surf + ci_surf, alpha=0.2)

        # Formatting both plots
        for idx, ax in enumerate(axes):
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Mould index")
            ax.set_title(titles[idx])
            ax.legend()
            ax.grid(True)

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()


def main():
    global NUM_SIMS, BATCH_SIZE, NO_HITS_MAX, TEMP, RH, ATTACH_PROB, DECAY_PROB
    test_3d()


if __name__ == "__main__":
    main()
