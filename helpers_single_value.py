'''
This file contains smaller functions needed in the main file 'dla_3d.py'
and 'dla_3dsim.py', these functions return a single value as output
'''
from numba import njit, prange
import numpy as np

@njit
def get_attach_prob(TEMP, RH):
    '''
    This function returns a float between 0 and 1, which denotes the probability of a
    particle attaching to the main cluster, based on inputted temperature and relative
    humidity
    '''
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


def get_decay_prob(ATTACH_PROB, decay_prob_multiplier, exp_dropoff):
    '''
    This function returns the decay probability as a float between 0 and 1 using
    exponential function, such that changes in attach probability are "felt more"
    '''
    return np.exp(-ATTACH_PROB * exp_dropoff) * decay_prob_multiplier


@njit(parallel=True)
def mold_coverage(grid, grid_size = 5):
    '''
    This function takes a grid and calculates how much mold covers the surface of
    the grid.
    '''
    # Uses grid sampling to smarter estimate mold coverage.
    # Divides the grid into 10x10 squares and counts the number of squares with mold.
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
                cell = grid[i * grid_size:(i + 1) * grid_size,
                            j * grid_size:(j + 1) * grid_size,
                            k * grid_size:(k + 1) * grid_size]

                # Check if there's any mold in the cell
                if np.any(cell > 0):
                    covered_cells += 1
    if covered_cells == 1:
        return 0

    return (covered_cells / total_cells) * 100


def coverage_to_m_value(cov):
    '''
    This function converts the given mold coverage into M-value
    '''
    return 14.87349 + (-0.03030586 - 14.87349)/(1 + (cov/271.0396)**0.4418942)

