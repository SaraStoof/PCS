import numpy as np 
from numba import jit, njit
import random
import matplotlib.pyplot as plt 
import math
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import plotly.graph_objects as go
from scipy.ndimage import convolve

# Constants
GRID_SIZE = 100
RADIUS = (GRID_SIZE // 2 ) + 5 # Radius of the circle
SEED = (GRID_SIZE // 2, GRID_SIZE // 2, GRID_SIZE)  # Seed in the middle of the grid
center_index = GRID_SIZE // 2
# Initialize grid (plus 1 to account for 0-index)
grid = np.zeros((GRID_SIZE + 1, GRID_SIZE + 1, GRID_SIZE + 1))
grid[center_index, center_index, center_index] = 1  # Set seed point as part of cluster

KERNEL = np.ones((3, 3, 3))

@njit
def conv3d(grid, kernel=KERNEL):

    d_grid, h_grid, w_grid = grid.shape
    d_kern, h_kern, w_kern = kernel.shape

    padding=0

    grid_temp = np.zeros((d_grid + 2, h_grid + 2, w_grid + 2))
    grid_temp[1:-1, 1:-1, 1:-1] = grid 

    new_grid = np.zeros((d_grid, h_grid, w_grid))
    for i in range(0, d_grid):
        for j in range(0, h_grid):
            for k in range(0, w_grid):
                # Get the 3D slice of the grid
                subgrid = grid_temp[i:i+d_kern, j:j+h_kern, k:k+w_kern]
                
                # Perform element-wise multiplication between the grid slice and kernel
                # Then sum up the result to get the corresponding output value
                new_grid[i, j, k] = np.sum(subgrid * kernel)

    return new_grid

@njit
def in_bounds(particles):
    return particles[
            (particles[:, 0] >= 0) & (particles[:, 0] < GRID_SIZE) &
            (particles[:, 1] >= 0) & (particles[:, 1] < GRID_SIZE) &
            (particles[:, 2] >= 0) & (particles[:, 2] < GRID_SIZE)
        ]

@njit
def move(particles):
    return np.random.randint(-1, 2, (len(particles), 3))

@njit
def check_neighbor(particles, grid, kernel=KERNEL):
    cluster_touch = conv3d(grid)  # The result of the convolution on the grid
    
    # List of indices to access the cluster_touch array
    indices = np.floor(particles)  # Ensure the indices are integers
    
    # Get values from cluster_touch using the particle positions (indices)
    values = cluster_touch[indices[:, 0], indices[:, 1], indices[:, 2]]

    # Now check which particles are connected to the cluster (values > 0)
    hits = particles[values > 0]
    
    return hits, cluster_touch


@njit
# This decorator tells Numba to compile this function using the JIT (just-in-time) compiler
def particle_loop(GRID_SIZE, RADIUS, grid, batch_size=10):
    touches_furthest_radius = False
    current_radius = 4 #spawns particles closer to where the seed is, to speed up the program. 
    particle_count = 0
    # and particle_count < 1500
    while touches_furthest_radius == False:  #keeps going until a particle touches the radius of the circle while being attached to the body
    # Create the particle starting from a random point on the circle
    
#         http://datagenetics.com/blog/january32020/index.html

        # theta = np.random.uniform(0, 2 * np.pi, batch_size)
        # phi = np.random.uniform(0, np.pi, batch_size)

        # particle = np.vstack([
        #     GRID_SIZE/2 + current_radius * np.sin(phi) * np.cos(theta),
        #     GRID_SIZE/2 + current_radius * np.sin(phi) * np.sin(theta),
        #     GRID_SIZE/2 + current_radius* np.cos(phi),
        # ]).astype(int).T

        theta = np.random.uniform(0, 2 * np.pi, batch_size)
        phi = np.random.uniform(0, np.pi, batch_size)

        # Initialize an empty array to hold the particle coordinates
        particle = np.zeros((batch_size, 3))

        # Populate the particle array manually
        particle[:, 0] = (GRID_SIZE / 2 + current_radius * np.sin(phi) * np.cos(theta))
        particle[:, 1] = (GRID_SIZE / 2 + current_radius * np.sin(phi) * np.sin(theta))
        particle[:, 2] = (GRID_SIZE / 2 + current_radius * np.cos(phi))

        # particle = (int(GRID_SIZE/2 + current_radius * math.sin(theta) * math.cos(phi)), 
        #             int(GRID_SIZE/2 + current_radius * math.sin(theta) * math.sin(phi)),
        #             int(GRID_SIZE/2 + current_radius* math.cos(theta))) #use angle and spawn point of seed (which is the middle of the grid) ...
        # ... to calculate the x and y coordinates of a new particle. Cast it to int also. 
        # particle_count += 1
        particle = in_bounds(particle)

        while len(particle) > 0:
            # Check if particle is out of bounds (ensure it's within grid size)
            # if min(particle) < 0 or max(particle) >= GRID_SIZE:
            #     break
            # particle = in_bounds(particle)

            particle += move(particle)

            particle = in_bounds(particle)

            hits, cluster_touch= check_neighbor(particle, grid)

            # Add particles that touched the cluster
            for hit in hits:
                grid[tuple(hit)] = 1
                dist_to_seed = np.linalg.norm(np.array(hit) - center_index)
                if dist_to_seed >= current_radius - 1:
                    current_radius += 5 
                    if current_radius > RADIUS:
                        touches_furthest_radius = True
            
            # Remove particles that hit the cluster
            # num_particles -= len(hits)
            particle = particle[
                cluster_touch[particle[:, 0], particle[:, 1], particle[:, 2]] == 0
            ]
    return


particle_loop(GRID_SIZE, RADIUS, grid)

x, y, z = np.where(grid == 1)

# Plot the 3D grid
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c='blue', s=GRID_SIZE//5, marker='s', linewidth=0)
# Set plot labels
ax.set_title("3D Particle Growth")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
