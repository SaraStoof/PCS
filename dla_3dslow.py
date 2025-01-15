import numpy as np 
from numba import njit
import random
import matplotlib.pyplot as plt 
import math
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import plotly.graph_objects as go

# Constants
GRID_SIZE = 1000
RADIUS = (GRID_SIZE // 2 ) + 5 # Radius of the circle
SEED = (GRID_SIZE // 2, GRID_SIZE // 2, GRID_SIZE)  # Seed in the middle of the grid
center_index = GRID_SIZE // 2
# Initialize grid (plus 1 to account for 0-index)
grid = np.zeros((GRID_SIZE + 1, GRID_SIZE + 1, GRID_SIZE + 1))
grid[center_index, center_index, center_index] = 1  # Set seed point as part of cluster

@njit # This decorator tells Numba to compile this function using the JIT (just-in-time) compiler
def particle_loop(GRID_SIZE, RADIUS, grid):
    touches_furthest_radius = False
    current_radius = 5 #spawns particles closer to where the seed is, to speed up the program. 
    particle_count = 0
    # and particle_count < 1500
    while touches_furthest_radius == False:  #keeps going until a particle touches the radius of the circle while being attached to the body
    # Create the particle starting from a random point on the circle
    
#         http://datagenetics.com/blog/january32020/index.html

        phi = random.uniform(0, 2 * math.pi) 
        theta = random.uniform(0, math.pi)
        particle = (int(GRID_SIZE/2 + current_radius * math.sin(theta) * math.cos(phi)), 
                    int(GRID_SIZE/2 + current_radius * math.sin(theta) * math.sin(phi)),
                    int(GRID_SIZE/2 + current_radius* math.cos(theta))) #use angle and spawn point of seed (which is the middle of the grid) ...
        # ... to calculate the x and y coordinates of a new particle. Cast it to int also. 
        particle_count += 1
        print(particle)

        while True:
            # Check if particle is out of bounds (ensure it's within grid size)
            if min(particle) < 0 or max(particle) >= GRID_SIZE:
                break
            
            # Check if the particle can attach to any adjacent grid cell (touches the cluster)
            if (grid[particle[0] + 1, particle[1], particle[2]] == 1 or
                grid[particle[0] - 1, particle[1], particle[2]] == 1 or
                grid[particle[0], particle[1] + 1, particle[2]] == 1 or
                grid[particle[0], particle[1] - 1, particle[2]] == 1 or
                grid[particle[0], particle[1], particle[2] + 1] == 1 or
                grid[particle[0], particle[1], particle[2] - 1] == 1        
               ):
                grid[particle[0], particle[1], particle[2]] = 1  # Attach particle to the grid

                dist_to_seed = math.sqrt((particle[0] - GRID_SIZE/2) ** 2 + (particle[1] - GRID_SIZE/2) ** 2 + (particle[2] - GRID_SIZE/2) ** 2)
                if dist_to_seed >= current_radius - 1:
                    current_radius += 5 
                    if current_radius > RADIUS:
                        touches_furthest_radius = True
                    #Stop the simulation if the particle touches the radius

                break  # Once attached, stop particle movement and move to the next particle

            # Move the particle randomly until we break the loop manually
            move = np.random.randint(0, 6)  # Randomly select one of four directions
            if move == 0:
                particle = (particle[0], particle[1] + 1,  particle[2])  # Move up
            elif move == 1:
                particle = (particle[0] + 1, particle[1], particle[2])  # Move right
            elif move == 2:
                particle = (particle[0], particle[1] - 1, particle[2])  # Move down
            elif move == 3:
                particle = (particle[0] - 1, particle[1], particle[2])  # Move left
            elif move == 4:
                particle = (particle[0], particle[1], particle[2] + 1)  # Move front
            elif move == 5:
                particle = (particle[0], particle[1], particle[2] - 1)  # Move back


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
