'''
This file contains the helper functions pertaining to user input
'''


def ask_grid_size(GRID_X, GRID_Y, GRID_Z, MAX_RADIUS):
    '''
    Handle user input for getting the size of the grid while checking for faulty inputs
    '''
    try:
        GRID_X = int(input("Enter max x-coordinate of grid: "))
        GRID_Y = int(input("Enter max y-coordinate of grid: "))
        GRID_Z = int(input("Enter max z-coordinate of grid: "))

        if GRID_X < 0 or GRID_Y < 0 or GRID_Z < 0:
            raise ValueError("Max coordinate can't be negative.")

    except (ValueError, TypeError):
        print("Invalid grid size. Defaulting to 100x100x100.")
        GRID_X, GRID_Y, GRID_Z = 100, 100, 100

    print("Grid size: ", GRID_X, GRID_Y, GRID_Z)
    MAX_RADIUS = (min(GRID_X, GRID_Y, GRID_Z) // 2) + 5
    return GRID_X, GRID_Y, GRID_Z, MAX_RADIUS


def ask_n_spawn_points():
    '''
    Handle user input for getting the size of the grid while checking for faulty inputs
    '''
    try:
        num_spawn_points = int(input("Enter how many random spawn points to initialize: "))

        if num_spawn_points <= 0:
            raise ValueError("Number of spawn points cannot be 0 or less.")

    except (ValueError, TypeError):
        print("Invalid number of spawn points. Defaulting to 5.")
        num_spawn_points = 5

    print("Number of spawn points: ", num_spawn_points)

    return num_spawn_points


def ask_spawn_point(GRID_X, GRID_Y, GRID_Z, MAX_RADIUS, SPAWN_ON_X_EDGE, SPAWN_ON_Y_EDGE, SPAWN_ON_Z_EDGE):
    '''
    Handle user input for getting the spawn point of the initial particle while checking
    for faulty inputs
    '''
    # Ask for spawn point
    try:

        SPAWN_X = int(input("Enter x-coordinate of spawn point: "))
        SPAWN_Y = int(input("Enter y-coordinate of spawn point: "))
        SPAWN_Z = int(input("Enter z-coordinate of spawn point: "))

        if SPAWN_X < 0 or SPAWN_X > GRID_X or SPAWN_Y < 0 or SPAWN_Y > GRID_Y or \
            SPAWN_Z < 0 or SPAWN_Z > GRID_Z or \
            not (SPAWN_X == 0 or SPAWN_X == GRID_X or SPAWN_Y == 0 or \
            SPAWN_Y == GRID_Y or SPAWN_Z == 0 or SPAWN_Z == GRID_Z):

            raise ValueError
    except (ValueError, TypeError):
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
    return SPAWN_X, SPAWN_Y, SPAWN_Z, MAX_RADIUS, SPAWN_ON_X_EDGE, SPAWN_ON_Y_EDGE, SPAWN_ON_Z_EDGE