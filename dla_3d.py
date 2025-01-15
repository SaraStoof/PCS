import numpy as np 
from numba import njit
import matplotlib.pyplot as plt 

# Constants
GRID_SIZE = 100
RADIUS = (GRID_SIZE // 2 ) + 5 # Radius of the circle
center_index = GRID_SIZE // 2

# Initialize grid (plus 1 to account for 0-index)
grid = np.zeros((GRID_SIZE + 1, GRID_SIZE + 1, GRID_SIZE + 1))
grid[center_index, center_index, center_index] = 1  # Set seed point as part of cluster

neighbor_offsets = np.array([
    [1, 0, 0], [-1, 0, 0],  # +x, -x
    [0, 1, 0], [0, -1, 0],  # +y, -y
    [0, 0, 1], [0, 0, -1]   # +z, -z
])

@njit
def in_bounds_mask(particles):
    return (
        (particles[:, 0] >= 0) & (particles[:, 0] < GRID_SIZE) &
        (particles[:, 1] >= 0) & (particles[:, 1] < GRID_SIZE) &
        (particles[:, 2] >= 0) & (particles[:, 2] < GRID_SIZE)
    )

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
def in_bounds(particles):
    return particles[
        (particles[:, 0] >= 0) & (particles[:, 0] < GRID_SIZE) &
        (particles[:, 1] >= 0) & (particles[:, 1] < GRID_SIZE) &
        (particles[:, 2] >= 0) & (particles[:, 2] < GRID_SIZE)
    ]


@njit
def move(particles):
    return particles + np.random.randint(-1, 2, (len(particles), 3))


@njit
def check_neighbor(particles, grid, batch_size):
    # numpy broadcasting
    neighbors = particles[:, None, :] + neighbor_offsets[None, :, :]
    neighbors = neighbors.reshape(-1, 3)

    # Get the valid mask for in-bounds neighbors
    mask = in_bounds_mask(neighbors)
    valid_neighbors = neighbors[mask]

    # Now match the original indices
    original_indices = np.nonzero(mask)[0]

    # Check if valid neighbors touch the grid
    hits_indices = []

    for idx, neighbor in enumerate(valid_neighbors):
        x, y, z = int(neighbor[0]), int(neighbor[1]), int(neighbor[2])
        if grid[x, y, z] == 1:
            hits_indices.append(original_indices[idx]) # Track original particle indices

    # Filter original particles by hits
    hits = [particles[i // 6] for i in hits_indices]
    p_indices = [i // 6 for i in hits_indices]

    # hits = []
    # mask = np.ones(len(particles), dtype=np.bool_)

    # for i in prange(len(hits_indices)):
    #     ind = hits_indices[i]
    #     p_index = ind // 6
    #     hits.append(particles[p_index])
    #     mask[p_index] = False


    # is_occupied = np.zeros(len(neighbors))

    # for i in range(len(neighbors)):
    #     x, y, z = int(neighbors[i, 0]), int(neighbors[i, 1]), int(neighbors[i, 2])
    #     if grid[x, y, z] == 1:
    #         is_occupied[i] = 1

    # hits_indices = np.where(is_occupied == 1)[0]

    # hits = [particles[i // 6] for i in hits_indices]

    return hits, p_indices

@njit
# This decorator tells Numba to compile this function using the JIT (just-in-time) compiler
def particle_loop(GRID_SIZE, RADIUS, grid, batch_size=1000):

    touches_furthest_radius = False
    current_radius = 4 #spawns particles closer to where the seed is, to speed up the program. 
    particle_count = 0

    while touches_furthest_radius == False:  #keeps going until a particle touches the radius of the circle while being attached to the body
    # Create the particle starting from a random point on the circle
    
    # http://datagenetics.com/blog/january32020/index.html

        theta = np.random.uniform(0, 2 * np.pi, batch_size)
        phi = np.random.uniform(0, np.pi, batch_size)

        # Initialize an empty array to hold the particle coordinates
        particle = np.zeros((batch_size, 3))

        # Populate the particle array manually
        particle[:, 0] = (GRID_SIZE / 2 + current_radius * np.sin(phi) * np.cos(theta))
        particle[:, 1] = (GRID_SIZE / 2 + current_radius * np.sin(phi) * np.sin(theta))
        particle[:, 2] = (GRID_SIZE / 2 + current_radius * np.cos(phi))

        particle = np.floor(particle)
        particle_count += batch_size

        particle = in_bounds(particle)

        print(particle_count)

        while len(particle) > 0:
            particle = move(particle)

            particle = in_bounds(particle)

            # check neighbors and update grid
            hits, p_indices = check_neighbor(particle, grid, batch_size)

            # Update grid
            for hit in hits:
                x, y, z = int(hit[0]), int(hit[1]), int(hit[2])
                grid[x, y, z] = 1
                dist_to_seed = np.linalg.norm(hit - center_index)
                if dist_to_seed >= current_radius - 1:
                    current_radius += 5 
                    if current_radius > RADIUS:
                        touches_furthest_radius = True
         
            # Remove particles that already attached themselves to the cluster
            particle = remove_indices(particle, p_indices)

    return


particle_loop(GRID_SIZE, RADIUS, grid)

x, y, z = np.where(grid == 1)

# Plot the 3D grid
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c='goldenrod', s=GRID_SIZE//5, marker='s', edgecolor='forestgreen')
# Set plot labels
ax.set_title("3D Particle Growth")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
