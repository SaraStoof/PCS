'''
This file contains the helper functions used for plotting purposes.
'''

def get_layer_count(grid, layer, axis=2):
    '''
    This function gets the amount of particles in a given layer
    '''
    count = 0
    axes = [0, 1, 2]
    axes.remove(axis)

    for i in range(grid.shape[axes[0]]):
        for j in range(grid.shape[axes[1]]):
            if grid[i, j, layer] > 0:
                count += 1
    return count

def get_grid_layer_counts(grid, axis=2):
    '''
    This function returns a list with every layer and the amount of particles in that
    layer
    '''
    layer_counts = []
    for i in range(grid.shape[axis]):
        layer_counts.append(get_layer_count(grid, i))
    return layer_counts