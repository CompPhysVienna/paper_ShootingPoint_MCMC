
import numpy as np
from numba import njit

@njit
def update_positions(current_x : float, force : float, dt_D : float, prefactor : float) -> float:
    # assumes beta = 1

    gauss_rand = np.random.randn()
    dx = dt_D * force + prefactor * gauss_rand
    
    current_x += dx

    return current_x 