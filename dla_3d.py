import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt


# Constants
GRID_SIZE = 100
RADIUS = (GRID_SIZE // 2) + 5  # Maximum radius of the circle
center_index = GRID_SIZE // 2
TIMESTEPS = 120
NUM_SIMS = 5
Temp = 30
RH = 97

# Initialize grid (plus 1 to account for 0-index)
# grid = np.zeros((GRID_SIZE + 1, GRID_SIZE + 1, GRID_SIZE + 1))
# grid[center_index, center_index, 0] = 1  # Set seed point as part of cluster

neighbor_offsets = np.array([
    [1, 0, 0], [-1, 0, 0],  # +x, -x
    [0, 1, 0], [0, -1, 0],  # +y, -y
    [0, 0, 1], [0, 0, -1]   # +z, -z
])


@njit
def attaching_prob(Temp, RH):
    RH_crit = (-0.00267 * (Temp**3)) + (0.16*(Temp**2)) - (3.13*Temp) + 100
    if(RH < RH_crit):
        return 0
    # The maximum M-value for the given temperature and relative humidity
    M_max = 1+7*((RH_crit-RH)/(RH_crit-100))-2*((RH_crit - RH)/(RH_crit-100))**2
    # The above two formulas are from the paper "A mathematical model of mould growth on
    # wooden material" by Hukka and Vitten 1999
    if(M_max < 0):
        return 0

    area_covered = 133.6561 + (0.9444885 - 133.6561)/(1 + (M_max/4.951036)**5.67479)
    # The formula for translating M-value to surface coverage represented by that
    # coverage, is retrieved by regression over the definition of M-value.
    # We use this as a stand-in for attachment probability.
    # The regression is over the points (0,0), (1,1), (3,10), (4,30), (5,70), (6,100)
    # These are the points where the M-value is 0, 1, 3, 4, 5, 6, respectively as given
    # by the table in "Development of an improved model for mould growth: Modelling"
    # by Viitanen et al. 2008
    if area_covered > 100:
        return 1
    return area_covered/500


def coverage_to_m_value(cov):
    return 14.87349 + (-0.03030586 - 14.87349)/(1 + (cov/271.0396)**0.4418942)


ATTACH_PROB = attaching_prob(Temp, RH)
DECAY_PROB = (1 - ATTACH_PROB) * 0.01


@njit(parallel=True)
def decay_grid(grid):
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
def in_bounds_neighbors(particles):
    return (
        (particles[:, 0] >= 0) & (particles[:, 0] <= GRID_SIZE) &
        (particles[:, 1] >= 0) & (particles[:, 1] <= GRID_SIZE) &
        (particles[:, 2] >= 0) & (particles[:, 2] <= GRID_SIZE)
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
def in_bounds(particles, radius):
    # if dist_to_seed >= radius + 5:
    return particles[
        (particles[:, 0] >= center_index - radius) & (particles[:, 0] < center_index + radius) &
        (particles[:, 1] >= center_index - radius) & (particles[:, 1] < center_index + radius) &
        (particles[:, 2] >= GRID_SIZE - radius) & (particles[:, 2] <= GRID_SIZE)
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
    mask = in_bounds_neighbors(neighbors)
    valid_neighbors = neighbors[mask]

    # Now match the original indices
    original_indices = np.nonzero(mask)[0]

    # Check if valid neighbors touch the grid
    hits_indices = []

    for idx, neighbor in enumerate(valid_neighbors):
        x, y, z = int(neighbor[0]), int(neighbor[1]), int(neighbor[2])
        if grid[x, y, z] == 1:

            depth = GRID_SIZE - z
            depth_bias_rate = 0.05
            depth_bias = np.exp(-depth_bias_rate * depth)
            # print("z:", z, "depth:", depth, "depth_bias:", depth_bias)

            if np.random.uniform() < ATTACH_PROB + depth_bias:
                # print("Attached")
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


# This decorator tells Numba to compile this function using the JIT (just-in-time) compiler
@njit
def particle_loop(grid, batch_size=1000):

    reached_edge = False
    # spawns particles closer to where the seed is, to speed up the program.
    current_radius = 5
    particle_count = 0

    # keeps going until a particle touches the radius of the circle while being attached to the body
    for i in range(TIMESTEPS):
        # Create the particle starting from a random point on the circle

        if i % int(TIMESTEPS*0.05) == 0:
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

        while len(particle) > 0:

            particle = move(particle)

            particle = in_bounds(particle, current_radius)

            # check neighbors and update grid
            hits, p_indices = check_neighbor(particle, grid, batch_size)

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

    return

@njit(parallel=True)
def monte_carlo():
    aggr_grid = np.zeros((GRID_SIZE + 1, GRID_SIZE + 1, GRID_SIZE + 1))
    for _ in prange(NUM_SIMS):
        # Initialize grid (plus 1 to account for 0-index)
        grid = np.zeros((GRID_SIZE + 1, GRID_SIZE + 1, GRID_SIZE + 1))
        grid[center_index, center_index, GRID_SIZE] = 1   # IMPORTANDT: REMOVED THE MINUS 1 KEEP LIKE THIS
        particle_loop(grid)

        aggr_grid += grid

    aggr_grid = aggr_grid/NUM_SIMS
    return aggr_grid

final_grid = monte_carlo()
mold_grid = final_grid.copy()
mold_grid[mold_grid > 0.02] = 1

mold_cov_3d = np.mean(mold_grid) * 100
mold_cov_surface = np.mean(mold_grid[:, :, GRID_SIZE]) * 100
#--- TEST PER LAYER HOW MANY PARTICLES ARE IN THE GRID ---

def check_layer(grid, layer):
    count = 0
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y, layer] > 0:
                count += 1
    return count

def check_grid(grid):
    layer_counts = []
    for z in range(grid.shape[2]):
        print("Layer", z, ":", check_layer(grid, z))
        layer_counts.append(check_layer(grid, z))
    return layer_counts


grid_layer_counts = check_grid(final_grid)
print(np.sum(grid_layer_counts))

# visualize grid_layer_counts in a plot
plt.plot(grid_layer_counts)
plt.xlabel("Layer")
plt.ylabel("Number of particles")
plt.title("Number of particles per layer")
plt.show()

# -------------------------------------------------------


print("attach_prob:", ATTACH_PROB)
print("decay_prob: ", DECAY_PROB)
print("Average mold coverage: ", mold_cov_3d, "%")
print("M-value: ", coverage_to_m_value(mold_cov_3d))
print("Average mold coverage surface: ", mold_cov_surface, "%")
print("M-value surface: ", coverage_to_m_value(mold_cov_surface))
print("Temperature: ", Temp)
print("Relative Humidity: ", RH)

# final_grid = np.zeros((GRID_SIZE + 1, GRID_SIZE + 1, GRID_SIZE + 1))
# final_grid[center_index, center_index, GRID_SIZE - 1] = 1
# particle_loop(final_grid)

# Plot the upper slice of the mold.
plt.imshow(final_grid[:, :, GRID_SIZE], cmap='Greens', interpolation='nearest')
plt.show()

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
