'''
The main file for the simulation, uses the other helper functions to run the simulations
'''

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import time
import sys
from helpers.helpers_single_value import *
from helpers.helpers_plots import *
from helpers.helpers_loop import *

TIMESTEPS_PER_DAY = 20

# Constants
NO_HITS_MAX = 5
BATCH_SIZE = 1000


@njit(parallel=True)
def decay_grid(grid, decay_prob):
    '''
    Introduces decay to the grid, setting certain particles to die by prioritizing
    particles around the edge of the cluster
    '''
    decay_amount = 0
    sum_grid = int(np.sum(grid))
    for _ in prange(sum_grid):
        if np.random.uniform() < decay_prob:
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
def particle_loop(grid, grid_x, grid_y, grid_z, spawn_x, spawn_y, spawn_z,
                  num_sims, days, temp, rh, attach_prob, decay_prob):
    '''
    This is the main loop of the simulation, it runs the simulation for mold growth for
    a certain amount of timesteps by making changes to the inputted grid
    '''
    reached_edge = False

    DAYS = 60
    timesteps = days * TIMESTEPS_PER_DAY

    # spawns particles closer to where the seed is, to speed up the program.
    current_radius = 5
    particle_count = 0

    spawn_on_x_edge = (spawn_x == 0, spawn_x == grid_x)
    spawn_on_y_edge = (spawn_y == 0, spawn_y == grid_y)
    spawn_on_z_edge = (spawn_z == 0, spawn_z == grid_z)

    max_radius = (min(grid_x, grid_y, grid_z) // 2) + 5

    for i in prange(timesteps):
        if i % TIMESTEPS_PER_DAY == 0:
            #These things happen once a day
            decay_grid(grid, decay_prob)

        # http://datagenetics.com/blog/january32020/index.html

        # Theta is the angle in the x-y plane
        # Phi is the angle from the z-axis
        theta = np.random.uniform(0, np.pi * 2, BATCH_SIZE)
        phi = np.random.uniform(np.pi, 2 * np.pi, BATCH_SIZE)

        # Initialize an empty array to hold the particle coordinates
        particles = np.zeros((BATCH_SIZE, 3))

        if reached_edge == False:
            # Populate the particle array manually
            particles[:, 0] = new_x_coords(theta, phi, current_radius, spawn_on_x_edge, spawn_x)
            particles[:, 1] = new_y_coords(theta, phi, current_radius, spawn_on_y_edge, spawn_y)
            particles[:, 2] = new_z_coords(phi, current_radius, spawn_on_z_edge, spawn_z)

        else:
            particles[:, 0] = (np.random.randint(0, grid_x, BATCH_SIZE))
            particles[:, 1] = (np.random.randint(0, grid_y, BATCH_SIZE))
            particles[:, 2] = (np.random.randint(0, grid_z, BATCH_SIZE))

        if len(particles) > current_radius**3:
            particles = particles[:current_radius**3]

        particles = np.floor(particles)
        particle_count += len(particles)

        particles = in_bounds(particles, current_radius, spawn_x, spawn_y, spawn_z,
                              spawn_on_x_edge, spawn_on_y_edge, spawn_on_z_edge)

        no_hits_count = 0

        while len(particles) > 0:

            particles = move(particles)

            particles = in_bounds(particles, current_radius, spawn_x, spawn_y, spawn_z,
                                  spawn_on_x_edge, spawn_on_y_edge, spawn_on_z_edge)


            # check neighbors and update grid
            hits, p_indices = check_neighbor(particles, grid, grid_x, grid_y, grid_z, attach_prob)

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
                dist_to_seed = np.linalg.norm(hit - np.array([spawn_x, spawn_y, spawn_z]))
                if dist_to_seed >= current_radius - 1 and reached_edge == False:
                    current_radius += 5
                    if current_radius >= max_radius:
                        reached_edge = True

            # Remove particles that already attached themselves to the cluster
            particles = remove_indices(particles, p_indices)






@njit
def loop_step(reached_edge, grid, particles, current_radius,
              grid_x, grid_y, grid_z,
              spawn_x, spawn_y, spawn_z, spawn_on_x_edge, spawn_on_y_edge, spawn_on_z_edge,
              attach_prob, MAX_RADIUS
              ):
    '''
    Performs one step in the loop, returns the updated current_radius and a boolean
    checking if the edge has been reached
    '''
    no_hits_count = 0
    while len(particles) > 0:

        particles = move(particles)

        particles = in_bounds(particles, current_radius,
                              spawn_x, spawn_y, spawn_z,
                              spawn_on_x_edge, spawn_on_y_edge, spawn_on_z_edge)


        # check neighbors and update grid
        hits, p_indices = check_neighbor(particles, grid, grid_x, grid_y, grid_z, attach_prob)

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
            dist_to_seed = np.linalg.norm(hit - np.array([spawn_x, spawn_y, spawn_z]))
            if dist_to_seed >= current_radius - 1 and reached_edge == False:
                current_radius += 5
                if current_radius >= MAX_RADIUS:
                    reached_edge = True

        # Remove particless that already attached themselves to the cluster
        particles = remove_indices(particles, p_indices)
    return current_radius, reached_edge



def one_batch_step(t, grid, temp, rh,
                   grid_x, grid_y, grid_z,
                   spawn_x, spawn_y, spawn_z,
                   spawn_on_x_edge, spawn_on_y_edge, spawn_on_z_edge,
                   current_radius=5, reached_edge=False, batch_size=1000):
    '''
    This is the main loop of the simulation, it runs the simulation for mold growth for
    a certain amount of timesteps by making changes to the inputted grid
    '''
    attach_prob = get_attach_prob(temp, rh)
    decay_prob = get_decay_prob(attach_prob, 0.05, 10)
    max_radius = (min(grid_x, grid_y, grid_z) // 2) + 5


    if t % TIMESTEPS_PER_DAY == 0:
        decay_grid(grid, decay_prob)

    theta = np.random.uniform(0, np.pi * 2, batch_size)
    phi = np.random.uniform(np.pi, 2 * np.pi, batch_size)

    # Initialize an empty array to hold the particles coordinates
    particles = np.zeros((batch_size, 3))

    if reached_edge == False:
        particles[:, 0] = new_x_coords(theta, phi, current_radius, spawn_on_x_edge, spawn_x)
        particles[:, 1] = new_y_coords(theta, phi, current_radius, spawn_on_y_edge, spawn_y)
        particles[:, 2] = new_z_coords(phi, current_radius, spawn_on_z_edge, spawn_z)

    else:
        particles[:, 0] = (np.random.randint(0, grid_x, batch_size))
        particles[:, 1] = (np.random.randint(0, grid_y, batch_size))
        particles[:, 2] = (np.random.randint(0, grid_z, batch_size))

    if len(particles) > current_radius**3:
        particles = particles[:current_radius**3]

    particles = np.floor(particles)

    particles = in_bounds(particles, current_radius, spawn_x, spawn_y, spawn_z,
                          spawn_on_x_edge, spawn_on_y_edge, spawn_on_z_edge)


    current_radius, reached_edge = \
        loop_step(reached_edge, grid, particles, current_radius,
                  grid_x, grid_y, grid_z,
                  spawn_x, spawn_y, spawn_z, spawn_on_x_edge, spawn_on_y_edge, spawn_on_z_edge,
                  attach_prob, max_radius)

    return current_radius, reached_edge, grid


@njit(parallel=True)
def monte_carlo(grid_x, grid_y, grid_z,
                spawn_x, spawn_y, spawn_z,
                num_sims, days, temp, rh, attach_prob, decay_prob):
    '''
    This function runs the simulation a certain amount of times and returns the average
    grid and mold coverage
    '''
    aggr_grid = np.zeros((grid_x + 1, grid_y + 1, grid_z + 1))
    mold_cov = 0
    for _ in prange(num_sims):
        # Initialize grid (plus 1 to account for 0-index)
        grid = np.zeros((grid_x + 1, grid_y + 1, grid_z + 1))
        grid[spawn_x, spawn_y, spawn_z] = 1
        particle_loop(grid, grid_x, grid_y, grid_z, spawn_x, spawn_y, spawn_z,
                      num_sims, days, temp, rh, attach_prob, decay_prob)

        aggr_grid += grid
        mold_cov += mold_coverage(grid)

    aggr_grid = aggr_grid / num_sims
    mold_cov = mold_cov / num_sims
    return aggr_grid, mold_cov


def run_for_webgui(grid_x, grid_y, grid_z,
                   spawn_x, spawn_y, spawn_z,
                   num_sims, days, temp, rh):
    print("Running for webgui")

    attach_prob = get_attach_prob(temp, rh)
    decay_prob = get_decay_prob(attach_prob, 0.05, 10)

    return monte_carlo(grid_x, grid_y, grid_z, spawn_x, spawn_y, spawn_z,
                       num_sims, days, temp, rh, attach_prob, decay_prob)
