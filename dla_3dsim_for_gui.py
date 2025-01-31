'''
The other main file for the simulation, uses the other helper functions to run the
simulations put displays every timestep in realtime
'''

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import time
import sys
from helpers.helpers_single_value import *
from helpers.helpers_plots import *
from helpers.helpers_loop import *
from helpers.helpers_user_input import *

TIMESTEPS_PER_DAY = 20

GRID_X, GRID_Y, GRID_Z = 100, 100, 100  # The max index of each axis
SPAWN_X, SPAWN_Y, SPAWN_Z = 50, 50, 100  # Ask user. Limit to surface point
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
                               (coords[i][1] - middle[1]) ** 2 + (coords[i][2] - middle[2]) ** 2)
    for _ in range(decay_amount):
        idx = np.argmax(distances)
        furthest = coords[idx]
        grid[int(furthest[0]), int(furthest[1]), int(furthest[2])] = 0
        distances[idx] = -1


@njit
def loop_step(reached_edge, grid, particles, current_radius, no_hits_count=0):
    '''
    Performs one step in the loop, returns the updated current_radius and a boolean
    checking if the edge has been reached
    '''
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

        # Remove particless that already attached themselves to the cluster
        particles = remove_indices(particles, p_indices)
    return current_radius, reached_edge



def one_batch_step(t, grid, sim_num, current_radius=5, reached_edge=False, batch_size=1000):
    '''
    This is the main loop of the simulation, it runs the simulation for mold growth for
    a certain amount of timesteps by making changes to the inputted grid
    '''

    if t % TIMESTEPS_PER_DAY == 0:
        decay_grid(grid)

    theta = np.random.uniform(0, np.pi * 2, batch_size)
    phi = np.random.uniform(np.pi, 2 * np.pi, batch_size)

    # Initialize an empty array to hold the particles coordinates
    particles = np.zeros((batch_size, 3))

    if reached_edge == False:
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

    particles = in_bounds(particles, current_radius, SPAWN_X, SPAWN_Y, SPAWN_Z,
                            SPAWN_ON_X_EDGE, SPAWN_ON_Y_EDGE, SPAWN_ON_Z_EDGE)

    current_radius, reached_edge = loop_step(reached_edge, grid, particles, current_radius)

    return current_radius, reached_edge, grid



def monte_carlo():
    '''
    This function runs the simulation a certain amount of times and returns the average
    grid and mold coverage
    '''
    aggr_grid = np.zeros((GRID_X + 1, GRID_Y + 1, GRID_Z + 1))
    mold_cov_3d = 0
    mold_cov_surf = 0

    for i in prange(NUM_SIMS):
        # Initialize grid (plus 1 to account for 0-index)
        grid = np.zeros((GRID_X + 1, GRID_Y + 1, GRID_Z + 1))
        grid[SPAWN_X, SPAWN_Y, SPAWN_Z] = 1
        particle_loop(grid, i, BATCH_SIZE)

        aggr_grid += grid
        mold_cov_3d += mold_coverage(grid)

        mold_cov_surf += (mold_cov_surface(grid[:, :, GRID_Z]) + mold_cov_surface(grid[:, :, 0]) + mold_cov_surface(grid[GRID_Z, :, :])
                          + mold_cov_surface(grid[0, :, :]) + mold_cov_surface(grid[:, GRID_Y, :]) + mold_cov_surface(grid[:, 0, :]))

    aggr_grid = aggr_grid/NUM_SIMS
    mold_cov_3d = mold_cov_3d / NUM_SIMS
    mold_cov_surf = mold_cov_surf/ NUM_SIMS
    return aggr_grid, mold_cov_3d, mold_cov_surf


