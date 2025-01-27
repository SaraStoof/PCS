'''
This file contains the helper functions used in the main loop in the main file
'''
from numba import njit, prange
import numpy as np

neighbor_offsets = np.array([
    [1, 0, 0], [-1, 0, 0],  # +x, -x
    [0, 1, 0], [0, -1, 0],  # +y, -y
    [0, 0, 1], [0, 0, -1]   # +z, -z
])

@njit
def in_bounds_neighbors(particles, GRID_X, GRID_Y, GRID_Z):
    return (
        (particles[:, 0] >= 0) & (particles[:, 0] <= GRID_X) &
        (particles[:, 1] >= 0) & (particles[:, 1] <= GRID_Y) &
        (particles[:, 2] >= 0) & (particles[:, 2] <= GRID_Z)
    )


@njit
def in_bounds(particles, radius, SPAWN_X, SPAWN_Y, SPAWN_Z,
              SPAWN_ON_X_EDGE, SPAWN_ON_Y_EDGE, SPAWN_ON_Z_EDGE):
    return particles[
        (particles[:, 0] >= SPAWN_X - radius * (not(SPAWN_ON_X_EDGE[0]))) &
        (particles[:, 0] <= SPAWN_X + radius * (not(SPAWN_ON_X_EDGE[1]))) &
        (particles[:, 1] >= SPAWN_Y - radius * (not(SPAWN_ON_Y_EDGE[0]))) &
        (particles[:, 1] <= SPAWN_Y + radius * (not(SPAWN_ON_Y_EDGE[1]))) &
        (particles[:, 2] >= SPAWN_Z - radius * (not(SPAWN_ON_Z_EDGE[0]))) &
        (particles[:, 2] <= SPAWN_Z + radius * (not(SPAWN_ON_Z_EDGE[1])))
    ]


@njit
def move(particles):
    return particles + np.random.randint(-1, 2, (len(particles), 3))


@njit
def dist_to_surface(x, y, z, GRID_X, GRID_Y, GRID_Z):
	dists = [x, y, z, GRID_X - x, GRID_Y - y, GRID_Z - z]
	return min(dists)


@njit
def check_neighbor(particles, grid, GRID_X, GRID_Y, GRID_Z, ATTACH_PROB):

    # numpy broadcasting
    neighbors = particles[:, None, :] + neighbor_offsets[None, :, :]
    neighbors = neighbors.reshape(-1, 3)

    # Get the valid mask for in-bounds neighbors
    mask = in_bounds_neighbors(neighbors, GRID_X, GRID_Y, GRID_Z)
    valid_neighbors = neighbors[mask]

    # Now match the original indices
    original_indices = np.nonzero(mask)[0]

    # Check if valid neighbors touch the grid
    hits_indices = []

    for idx, neighbor in enumerate(valid_neighbors):
        x, y, z = int(neighbor[0]), int(neighbor[1]), int(neighbor[2])
        if grid[x, y, z] == 1:

            depth = dist_to_surface(x, y, z, GRID_X, GRID_Y, GRID_Z)
            depth_bias_rate = 0.05
            depth_bias = np.exp(-depth_bias_rate * depth)
            # print("x:", x, "y:", y, "z:", z, "depth:", depth, "depth_bias:", depth_bias)

            if np.random.uniform() < ATTACH_PROB + depth_bias:
                # Track original particle indices
                hits_indices.append(original_indices[idx])

    # Filter original particles by hits
    hits = [particles[i // 6] for i in hits_indices]
    p_indices = [i // 6 for i in hits_indices]

    return hits, p_indices

@njit
def nonneg_arr(arr):
    # Flattens all negative values to 0. Makes the array nonnegative.
    arr[np.where(arr < 0.0)] = 0
    return arr

@njit
def remove_indices(arr, indices_to_remove):
    # Create a mask to keep all elements by default
    mask = np.ones(len(arr), dtype=np.bool_)

    # Mark indices to remove as False
    for idx in indices_to_remove:
        mask[idx] = False

    # Filter the array using the mask
    return arr[mask]


@njit
def new_x_coords(theta, phi, current_radius, SPAWN_ON_X_EDGE, SPAWN_X):
    '''
    This function returns lists of coordinates for the new batch of particles
    based on the spawn point and the current radius.
    When the coordinate falls outside of the grid, it defaults to the edge of the grid.
    which gives the surface twice the chance of being hit.
    '''
    if SPAWN_ON_X_EDGE[1]:
        return (SPAWN_X - nonneg_arr(current_radius * np.sin(phi) * np.cos(theta)))
    elif SPAWN_ON_X_EDGE[0]:
        return (SPAWN_X + nonneg_arr(current_radius * np.sin(phi) * np.cos(theta)))
    return (SPAWN_X + current_radius * np.sin(phi) * np.cos(theta))


@njit
def new_y_coords(theta, phi, current_radius, SPAWN_ON_Y_EDGE, SPAWN_Y):
    if SPAWN_ON_Y_EDGE[1]:
        return (SPAWN_Y - nonneg_arr(current_radius * np.sin(phi) * np.sin(theta)))
    elif SPAWN_ON_Y_EDGE[0]:
        return (SPAWN_Y + nonneg_arr(current_radius * np.sin(phi) * np.sin(theta)))
    return (SPAWN_Y + current_radius * np.sin(phi) * np.sin(theta))


@njit
def new_z_coords(phi, current_radius, SPAWN_ON_Z_EDGE, SPAWN_Z):
    if SPAWN_ON_Z_EDGE[1]:  # If spawn point is on the top edge
        return (SPAWN_Z - nonneg_arr(current_radius * np.cos(phi)))
    elif SPAWN_ON_Z_EDGE[0]:  # If spawn point is on the bottom edge
        return (SPAWN_Z + nonneg_arr(current_radius * np.cos(phi)))
    return (SPAWN_Z + current_radius * np.cos(phi))
