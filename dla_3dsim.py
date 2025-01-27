'''
The other main file for the simulation, uses the other helper functions to run the
simulations put displays every timestep in realtime
'''

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from helpers.helpers_single_value import *
from helpers.helpers_plots import *
from helpers.helpers_loop import *

# Constants
GRID_SIZE = 100
RADIUS = (GRID_SIZE // 2) + 5  # Maximum radius of the circle
center_index = GRID_SIZE // 2
TIMESTEPS = 120
NUM_SIMS = 5
TEMP = 30
RH = 97


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
def loop_step(reached_edge, grid, particle, current_radius, no_hits_count):
    '''
    Performs one step in the loop, returns the updated current_radius and a boolean
    checking if the edge has been reached
    '''
    while len(particle) > 0:

        particle = move(particle)

        particle = in_bounds(particle, current_radius)

        # check neighbors and update grid
        hits, p_indices = check_neighbor(particle, grid)

        # Break if particles have moved five turns with no hits.
        if len(hits) == 0:
            no_hits_count += 1
            if no_hits_count > 5:
                break
        else:
            no_hits_count = 0

        # Update grid
        for hit in hits:
            x, y, z = int(hit[0]), int(hit[1]), int(hit[2])
            grid[x, y, z] = 1
            dist_to_seed = np.linalg.norm(hit - np.array([center_index, center_index, GRID_SIZE]))
            if dist_to_seed >= current_radius - 1 and reached_edge == False:
                current_radius += 5
                if current_radius >= RADIUS:
                    reached_edge = True

        # Remove particles that already attached themselves to the cluster
        particle = remove_indices(particle, p_indices)
    return current_radius, reached_edge


# This decorator tells Numba to compile this function using the JIT (just-in-time) compiler
def particle_loop(grid, sim_num, batch_size=1000):
    '''
    This is the main loop of the simulation, it runs the simulation for mold growth for
    a certain amount of timesteps by making changes to the inputted grid
    '''
    reached_edge = False
    # spawns particles closer to where the seed is, to speed up the program.
    current_radius = 5
    particle_count = 0

    for t in range(TIMESTEPS):
        ax.clear()
        x, y, z = np.where(grid > 0)
        ax.scatter(x, y, z, c='goldenrod', s=GRID_SIZE //
                   5, marker='s', edgecolor='forestgreen')
        ax.set_title(f"3D Mold Growth - Timestep {t+1}, Simulation number {sim_num+1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.draw()
        plt.pause(0.1)
        if t % int(TIMESTEPS*0.05) == 0:
            decay_grid(grid)

        # http://datagenetics.com/blog/january32020/index.html

        # Theta is the angle in the x-y plane
        # Phi is the angle from the z-axis
        theta = np.random.uniform(0, np.pi * 2, batch_size)
        phi = np.random.uniform(np.pi, 2 * np.pi, batch_size)

        # Initialize an empty array to hold the particle coordinates
        particle = np.zeros((batch_size, 3))

        if reached_edge == False:
            # Populate the particle array manually
            particle[:, 0] = (center_index + current_radius *
                              np.sin(phi) * np.cos(theta))
            particle[:, 1] = (center_index + current_radius *
                              np.sin(phi) * np.sin(theta))
            particle[:, 2] = (GRID_SIZE -
                              nonneg_arr(current_radius * np.cos(phi)))

        else:
            particle[:, 0] = (np.random.randint(0, GRID_SIZE, batch_size))
            particle[:, 1] = (np.random.randint(0, GRID_SIZE, batch_size))
            particle[:, 2] = (np.random.randint(0, GRID_SIZE, batch_size))

        if len(particle) > current_radius**3:
            particle = particle[:current_radius**3]

        particle = np.floor(particle)
        particle_count += len(particle)

        particle = in_bounds(particle, current_radius)

        no_hits_count = 0

        current_radius, reached_edge = loop_step(reached_edge, grid, particle, current_radius, no_hits_count)

    return

# Initialize Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def monte_carlo():
    '''
    This function runs the simulation a certain amount of times and returns the average
    grid and mold coverage
    '''
    aggr_grid = np.zeros((GRID_SIZE + 1, GRID_SIZE + 1, GRID_SIZE + 1))
    for i in range(NUM_SIMS):
        # Initialize grid (plus 1 to account for 0-index)
        grid = np.zeros((GRID_SIZE + 1, GRID_SIZE + 1, GRID_SIZE + 1))
        grid[center_index, center_index, GRID_SIZE] = 1   # IMPORTANDT: REMOVED THE MINUS 1 KEEP LIKE THIS
        particle_loop(grid, i)

        aggr_grid += grid

    aggr_grid = aggr_grid/NUM_SIMS
    return aggr_grid

final_grid = monte_carlo()
mold_grid = final_grid.copy()
mold_grid[mold_grid > 0.02] = 1

mold_cov_3d = np.mean(mold_grid) * 100
mold_cov_surface = np.mean(mold_grid[:, :, GRID_SIZE]) * 100
#--- TEST PER LAYER HOW MANY PARTICLES ARE IN THE GRID ---
grid_layer_counts = get_grid_layer_counts(final_grid)
print(np.sum(grid_layer_counts))

# visualize grid_layer_counts in a plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(grid_layer_counts)
ax.set_xlabel("Layer")
ax.set_ylabel("Number of particles")
ax.set_title("Number of particles per layer")
plt.draw()

# -------------------------------------------------------

ATTACH_PROB = get_attach_prob(TEMP, RH)
DECAY_PROB = get_decay_prob(ATTACH_PROB, 0.05, 10)
print("attach_prob:", ATTACH_PROB)
print("decay_prob: ", DECAY_PROB)
print("Average mold coverage: ", mold_cov_3d, "%")
print("M-value: ", coverage_to_m_value(mold_cov_3d))
print("Average mold coverage surface: ", mold_cov_surface, "%")
print("M-value surface: ", coverage_to_m_value(mold_cov_surface))
print("Temperature: ", TEMP)
print("Relative Humidity: ", RH)

# Plot the upper slice of the mold.
fig, ax = plt.subplots(figsize=(6, 6))  # Ensure a 2D context for imshow
ax.imshow(final_grid[:, :, GRID_SIZE], cmap='Greens', interpolation='nearest')
ax.set_title("Middle Slice of the Mold")
plt.draw()

x, y, z = np.where(final_grid >= 1 / NUM_SIMS)

# Plot the 3D grid
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c='goldenrod', s=GRID_SIZE //
                     5, marker='s', edgecolor='forestgreen')

# Set plot labels
ax.set_title("3D Mold Growth")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
