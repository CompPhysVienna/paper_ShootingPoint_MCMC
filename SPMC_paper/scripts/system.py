
import numpy as np
from numba import njit

@njit
def potential_energy_function(x : float) -> float:

    barrier_height = 5
    x_pot = x - 1

    if x_pot < 0:
        U = x_pot**2 * (0.01 * x_pot**2 - 1)
    else:
        U = x_pot**2 * (0.16 * x_pot**2 - 4)

    return barrier_height/25 * U

@njit
def force_function(x : float) -> float:

    barrier_height = 5
    x_pot = x - 1

    if x_pot < 0:
        F = -(x_pot*(x_pot**2-50))/25
    else:
        F = -(16*x_pot**3-200*x_pot)/25

    return barrier_height/25 * F


@njit
def check_reactive(proposal_path : np.ndarray, state_A_boundary : float, state_B_boundary : float):

    start_in_A = proposal_path[0] < state_A_boundary
    start_in_B = proposal_path[0] > state_B_boundary
    end_in_A = proposal_path[-1] < state_A_boundary
    end_in_B = proposal_path[-1] > state_B_boundary

    is_reactive_path = (start_in_A and end_in_B) or (start_in_B and end_in_A)

    return is_reactive_path