import numpy as np
from numba import njit

from scripts.integrator import update_positions


@njit
def generate_path_to_state(shooting_point  : float,
                           state_A_boundary : float, state_B_boundary : float, 
                           dt_D : float, force_function : callable) -> np.ndarray:
    """
    Generate a path from shooting_point to either state_A_boundary or state_B_boundary.
    For simplicity, we write out every frame and have no maximum path length.
    """

    previous_x = shooting_point
    prefactor = np.sqrt(2*dt_D)

    trajectory = []
    
    while True:

        current_force = force_function(previous_x)
        previous_x = update_positions(previous_x, current_force, dt_D, prefactor)

        trajectory.append(previous_x)

        if previous_x < state_A_boundary or previous_x > state_B_boundary:
            return np.array(trajectory)
        

@njit
def shoot_two_way_flexible(shooting_point  : float,
                           state_A_boundary : float, state_B_boundary : float, 
                           dt_D : float, force_function : callable) -> np.ndarray:
    """
    Perform a two-way shooting move from shooting_point to either state_A_boundary or state_B_boundary.
    
    This is again a minimal code, usually one would need to check that the shooting point is not in a 
    stable state, since then only a forward trajectory would need to be generated. As stated above, we also
    don't set a maximum path length.
    """

    # Generate forward and backward trajectory
    forward_trj = generate_path_to_state(shooting_point, state_A_boundary, state_B_boundary, dt_D, force_function)
    reverse_trj = generate_path_to_state(shooting_point, state_A_boundary, state_B_boundary, dt_D, force_function)

    proposal_path = np.hstack((reverse_trj[::-1], np.array([shooting_point]), forward_trj))
    
    # Save index of shooting point on new path
    proposal_sp_index = len(reverse_trj)
    
    return proposal_path, proposal_sp_index


@njit
def shoot_one_way_flexible(current_path : np.ndarray, sp_index : int, direction : int,
                           state_A_boundary : float, state_B_boundary : float, 
                           dt_D : float, force_function : callable) -> np.ndarray:
    """
    Perform a one-way shooting move from shooting_point to either state_A_boundary or state_B_boundary.
    
    As stated above, we don't check if the shooting point is in a stable state and don't set 
    a maximum path length.
    """

    shooting_point = current_path[sp_index]

    # Generate forward and backward trajectory
    partial_trj = generate_path_to_state(shooting_point, state_A_boundary, state_B_boundary, dt_D, force_function)
    
    if direction == -1:
        reverse_trj = current_path[:sp_index+1].copy()
        forward_trj = partial_trj
        
    else:
        reverse_trj = partial_trj[::-1]
        forward_trj = current_path[sp_index:].copy()

    proposal_path = np.hstack((reverse_trj, forward_trj))
    proposal_sp_index = len(reverse_trj) if direction == 1 else len(reverse_trj) - 1

    return proposal_path, proposal_sp_index
