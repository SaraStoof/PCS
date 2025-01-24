import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import time
import sys

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

neighbor_offsets = np.array([
    [1, 0, 0], [-1, 0, 0],  # +x, -x
    [0, 1, 0], [0, -1, 0],  # +y, -y
    [0, 0, 1], [0, 0, -1]   # +z, -z
])


@njit
def attaching_prob(TEMP, RH):
    RH_crit = (-0.00267 * (TEMP**3)) + (0.16*(TEMP**2)) - (3.13*TEMP) + 100
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
    return area_covered / 500


ATTACH_PROB = attaching_prob(TEMP, RH)
DECAY_PROB = (1 - ATTACH_PROB) * 0.01


def coverage_to_m_value(cov):
    return 14.87349 + (-0.03030586 - 14.87349)/(1 + (cov/271.0396)**0.4418942)


def get_decay_prob(decay_prob_multiplier, exponential_drop_off):
    #Using exponential function to calculate decay rate, such that changes in attach prob are "felt more"
    return np.exp(-ATTACH_PROB * exponential_drop_off) * decay_prob_multiplier


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
                               (coords[i][1] - middle[1]) ** 2 +
                               (coords[i][2] - middle[2]) ** 2)
    for _ in range(decay_amount):
        idx = np.argmax(distances)
        furthest = coords[idx]
        grid[int(furthest[0]), int(furthest[1]), int(furthest[2])] = 0
        distances[idx] = -1

@njit(parallel=True)
def mold_coverage(grid, grid_size = 5):
    #Uses grid sampling to smarter estimate mold coverage. Divides the grid into 10x10 squares and counts the number of squares with mold.
    height, width, depth = grid.shape
    cells_x = width // grid_size
    cells_y = height // grid_size
    cells_z = depth // grid_size

    covered_cells = 0
    total_cells = cells_x * cells_y * cells_z

    for i in prange(cells_y):
        for j in prange(cells_x):
            for k in prange(cells_y):
                # Extract grid cell
                cell = grid[i * grid_size:(i + 1) * grid_size, j * grid_size:(j + 1) * grid_size, k * grid_size:(k + 1) * grid_size ]

                # Check if there's any mold in the cell
                if np.any(cell > 0):
                    covered_cells += 1
    if covered_cells == 1:
        return 0

    return (covered_cells / total_cells) * 100

@njit
def in_bounds_neighbors(particles):
    return (
        (particles[:, 0] >= 0) & (particles[:, 0] <= GRID_X) &
        (particles[:, 1] >= 0) & (particles[:, 1] <= GRID_Y) &
        (particles[:, 2] >= 0) & (particles[:, 2] <= GRID_Z)
    )


@njit
def in_bounds(particles, radius):

    return particles[
        (particles[:, 0] >= SPAWN_X - radius * (not(SPAWN_ON_X_EDGE[0]))) & (particles[:, 0] <= SPAWN_X + radius * (not(SPAWN_ON_X_EDGE[1]))) &
        (particles[:, 1] >= SPAWN_Y - radius * (not(SPAWN_ON_Y_EDGE[0]))) & (particles[:, 1] <= SPAWN_Y + radius * (not(SPAWN_ON_Y_EDGE[1]))) &
        (particles[:, 2] >= SPAWN_Z - radius * (not(SPAWN_ON_Z_EDGE[0]))) & (particles[:, 2] <= SPAWN_Z + radius * (not(SPAWN_ON_Z_EDGE[1])))
    ]


@njit
def move(particles):
    return particles + np.random.randint(-1, 2, (len(particles), 3))

@njit
def dist_to_surface(x, y, z):
	dists = [x, y, z, GRID_X - x, GRID_Y - y, GRID_Z - z]
	return min(dists)

# -------------------

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

            depth = dist_to_surface(x, y, z)
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



#DONT FLATTEN IT, MAKE IT BE ON THE SURFACE MAX_RADIUS
@njit
def new_x_coords(theta, phi, current_radius):
    if SPAWN_ON_X_EDGE[1]:
        return (SPAWN_X - nonneg_arr(current_radius * np.sin(phi) * np.cos(theta)))
    elif SPAWN_ON_X_EDGE[0]:
        return (SPAWN_X + nonneg_arr(current_radius * np.sin(phi) * np.cos(theta)))
    return (SPAWN_X + current_radius * np.sin(phi) * np.cos(theta))


@njit
def new_y_coords(theta, phi, current_radius):
    if SPAWN_ON_Y_EDGE[1]:
        return (SPAWN_Y - nonneg_arr(current_radius * np.sin(phi) * np.sin(theta)))
    elif SPAWN_ON_Y_EDGE[0]:
        return (SPAWN_Y + nonneg_arr(current_radius * np.sin(phi) * np.sin(theta)))
    return (SPAWN_Y + current_radius * np.sin(phi) * np.sin(theta))


@njit
def new_z_coords(phi, current_radius):
    # Returns a list of z-coordinates for the new batch of particles.
    # The z-coordinates are based on the spawn point and the current radius.
    # When the coordinate falls outside of the grid, it defaults to the edge of the grid.
    # which gives the surface twice the chance of being hit.
    if SPAWN_ON_Z_EDGE[1]:  # If spawn point is on the top edge
        return (SPAWN_Z - nonneg_arr(current_radius * np.cos(phi)))
    elif SPAWN_ON_Z_EDGE[0]:  # If spawn point is on the bottom edge
        return (SPAWN_Z + nonneg_arr(current_radius * np.cos(phi)))
    return (SPAWN_Z + current_radius * np.cos(phi))


# This decorator tells Numba to compile this function using the JIT (just-in-time) compiler
@njit
def particle_loop(grid, batch_size=1000):
    reached_edge = False
    # spawns particles closer to where the seed is, to speed up the program.
    current_radius = 5
    particle_count = 0

    # keeps going until a particle touches the radius of the circle while being attached to the body
    for i in prange(TIMESTEPS):
        # Create the particle starting from a random point on the circle

        if i % TIMESTEPS_PER_DAY == 0:
            #These things happen once a day
            decay_grid(grid)

        # http://datagenetics.com/blog/january32020/index.html

        # Theta is the angle in the x-y plane
        # Phi is the angle from the z-axis
        theta = np.random.uniform(0, np.pi * 2, batch_size)
        phi = np.random.uniform(np.pi, 2 * np.pi, batch_size)

        # Initialize an empty array to hold the particle coordinates
        particles = np.zeros((batch_size, 3))

        if reached_edge == False:
            # Populate the particle array manually
            particles[:, 0] = new_x_coords(theta, phi, current_radius)
            particles[:, 1] = new_y_coords(theta, phi, current_radius)
            particles[:, 2] = new_z_coords(phi, current_radius)

        else:
            particles[:, 0] = (np.random.randint(0, GRID_X, batch_size))
            particles[:, 1] = (np.random.randint(0, GRID_Y, batch_size))
            particles[:, 2] = (np.random.randint(0, GRID_Z, batch_size))

        # Drop all particles that have spawned outside of the grid to the surface

        if len(particles) > current_radius**3:
            particles = particles[:current_radius**3]

        particles = np.floor(particles)
        particle_count += len(particles)

        particles = in_bounds(particles, current_radius)

        no_hits_count = 0

        while len(particles) > 0:

            particles = move(particles)

            particles = in_bounds(particles, current_radius)

            # check neighbors and update grid
            hits, p_indices = check_neighbor(particles, grid, batch_size)

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

@njit(parallel=True)
def monte_carlo():

    aggr_grid = np.zeros((GRID_X + 1, GRID_Y + 1, GRID_Z + 1))
    mold_cov = 0
    for i in prange(NUM_SIMS):
        # Initialize grid (plus 1 to account for 0-index)
        grid = np.zeros((GRID_X + 1, GRID_Y + 1, GRID_Z + 1))
        grid[SPAWN_X, SPAWN_Y, SPAWN_Z] = 1
        particle_loop(grid, BATCH_SIZE)

        aggr_grid += grid
        mold_cov += mold_coverage(grid)

    aggr_grid = aggr_grid/NUM_SIMS
    mold_cov = mold_cov / NUM_SIMS
    return aggr_grid, mold_cov

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
        # print("Layer", z, ":", check_layer(grid, z))
        layer_counts.append(check_layer(grid, z))
    return layer_counts

def visualize(final_grid, mold_cov_3d, mold_cov_surface, mold_cov_new):
    #--- TEST PER LAYER HOW MANY PARTICLES ARE IN THE GRID ---

    grid_layer_counts = check_grid(final_grid)
    # print(np.sum(grid_layer_counts))


    # visualize grid_layer_counts in a plot
    plt.plot(grid_layer_counts)
    plt.xlabel("Layer")
    plt.ylabel("Number of particles")
    plt.title("Number of particles per layer")
    plt.show()

    print("attach_prob:", ATTACH_PROB)
    print("decay_prob: ", DECAY_PROB)
    print("Average mold coverage: ", mold_cov_3d, "%")
    print("Average mold coverage new: ", mold_cov_new, "%")
    print("Average mold coverage surface: ", mold_cov_surface, "%")
    print("M-value: ", coverage_to_m_value(mold_cov_3d))
    print("M-value surface: ", coverage_to_m_value(mold_cov_surface))

    # Plot the upper slice of the mold.
    plt.imshow(final_grid[:, :, GRID_Z], cmap='Greens', interpolation='nearest')
    plt.show()

    plt.imshow(final_grid[:, GRID_Y // 2, :], cmap='Greens', interpolation='nearest')
    plt.show()

    x, y, z = np.where(final_grid >= 1 / NUM_SIMS)

    # Plot the 3D grid
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='goldenrod', s=GRID_X // 5,
               marker='s', edgecolor='forestgreen')

    # Set plot labels
    ax.set_title("3D Mold Growth")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def ask_spawn_point():
    # Ask for spawn point
    global SPAWN_X, SPAWN_Y, SPAWN_Z
    global SPAWN_ON_X_EDGE, SPAWN_ON_Y_EDGE, SPAWN_ON_Z_EDGE

    SPAWN_X = int(input("Enter x-coordinate of spawn point: "))
    SPAWN_Y = int(input("Enter y-coordinate of spawn point: "))
    SPAWN_Z = int(input("Enter z-coordinate of spawn point: "))

    if SPAWN_X < 0 or SPAWN_X > GRID_X or SPAWN_Y < 0 or SPAWN_Y > GRID_Y or \
       SPAWN_Z < 0 or SPAWN_Z > GRID_Z or \
       not (SPAWN_X == 0 or SPAWN_X == GRID_X or SPAWN_Y == 0 or \
            SPAWN_Y == GRID_Y or SPAWN_Z == 0 or SPAWN_Z == GRID_Z):
        print("Invalid spawn point. Defaulting to center of grid.")
        SPAWN_X = GRID_X // 2
        SPAWN_Y = GRID_Y // 2
        SPAWN_Z = GRID_Z

    SPAWN_ON_X_EDGE = (SPAWN_X == 0, SPAWN_X == GRID_X)
    SPAWN_ON_Y_EDGE = (SPAWN_Y == 0, SPAWN_Y == GRID_Y)
    SPAWN_ON_Z_EDGE = (SPAWN_Z == 0, SPAWN_Z == GRID_Z)

    print("Spawn point: ", SPAWN_X, SPAWN_Y, SPAWN_Z)
    print("On edge: ", SPAWN_ON_X_EDGE, SPAWN_ON_Y_EDGE, SPAWN_ON_Z_EDGE)
    print("Radius: ", MAX_RADIUS)



def main():
    global NUM_SIMS, BATCH_SIZE, NO_HITS_MAX, TEMP, RH, ATTACH_PROB, DECAY_PROB
    if len(sys.argv) == 5:
        NUM_SIMS = int(sys.argv[1])
        BATCH_SIZE = int(sys.argv[2])
        TEMP = int(sys.argv[3])
        RH = int(sys.argv[4])
    else:
        print("Not enough arguments. Defaulting to NUM_SIMS, BATCH_SIZE, TEMP, RH: ", NUM_SIMS, BATCH_SIZE, TEMP, RH)
    ATTACH_PROB = attaching_prob(TEMP, RH)
    DECAY_PROB = get_decay_prob(0.05, 10)

    ask_spawn_point()

    start = time.time()
    final_grid, mold_cov_new = monte_carlo()
    end = time.time()
    mold_grid = final_grid.copy()
    mold_grid[mold_grid > 0.02] = 1

    mold_cov_3d = np.mean(mold_grid) * 100
    mold_cov_surface = np.mean(mold_grid[:, :, GRID_Z]) * 100
    # print(NUM_SIMS, end - start, BATCH_SIZE, TIMESTEPS, NO_HITS_MAX, mold_cov_3d, mold_cov_surface, mold_cov_new)
    print(f"Time taken: {end - start}")

    visualize(final_grid, mold_cov_3d, mold_cov_surface, mold_cov_new)


if __name__ == "__main__":
    main()
