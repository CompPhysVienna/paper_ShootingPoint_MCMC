import numpy as np
from numba import njit

from scripts.integrator import update_positions

@njit
def generate_path(shooting_point  : float, N_steps : int,
                  dt_D : float, force_function : callable) -> np.ndarray:
  
    current_x = shooting_point
    prefactor = np.sqrt(2*dt_D)

    trajectory = np.empty(N_steps)
    
    for step in range(N_steps):

        current_force = force_function(current_x)
        current_x = update_positions(current_x, current_force, dt_D, prefactor)

        trajectory[step] = current_x

    return trajectory


@njit
def shoot_two_way_fixed_length(shooting_point : float, fw_steps : int, rv_steps : int,
                               dt_D : float, force_function : callable) -> np.ndarray:
  
    # Generate forward and backward trajectory
    forward_trj = generate_path(shooting_point, fw_steps, dt_D, force_function)
    reverse_trj = generate_path(shooting_point, rv_steps, dt_D, force_function)

    proposal_path = np.hstack((reverse_trj[::-1], np.array([shooting_point]), forward_trj))
        
    # Save index of shooting point on new path
    proposal_sp_index = len(reverse_trj)
    
    return proposal_path, proposal_sp_index