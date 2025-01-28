
from numba import njit
import numpy as np
import ray
import sys 

from scripts.system import force_function, check_reactive
from scripts.path_generation_fixed import shoot_two_way_fixed_length
from scripts.statistics import initialize_statistics, update_statistics


@njit
def _run_aimless_TPS_fixed_length(total_trials : int, initial_path : np.ndarray, 
                                  path_length : int, equilibration_trials : int, 
                                  state_bounds : float , dt_D : float, configuration_bins : np.ndarray, 
                                  length_bins : np.ndarray, k_bins : np.ndarray, max_dk : int):
    
    current_path = initial_path.copy()
    
    # Get the index of the path center
    center = current_path.shape[0] // 2
    current_sp_index = center
    
    # Possible indices to choose from
    indices = [center - max_dk, center + max_dk]
    
    # Statistics
    path_histogram, sp_histogram, length_histogram, relative_k_histogram = initialize_statistics(configuration_bins, length_bins, k_bins)
    
    
    for trial in range(total_trials):

        # 1st SHIFT
        sp_index = indices[np.random.randint(0, 2)]
        shooting_point = current_path[sp_index]

        # Generate forward and backward trajectory, can have different leg lengths depending on new SP index
        proposal_trj, new_sp_index = shoot_two_way_fixed_length(shooting_point, center - max_dk, center + max_dk, dt_D, force_function)

        # If it is reactive, accept
        is_reactive_path = check_reactive(proposal_trj, state_bounds[0], state_bounds[1])
        if is_reactive_path:

            current_path = proposal_trj

            # 2nd SHIFT
            current_sp_index = indices[np.random.randint(0, 2)]

        # Statistics
        if trial > equilibration_trials:
            path_histogram, sp_histogram, length_histogram, relative_k_histogram =  update_statistics(current_path, current_sp_index,
                                                                                                      path_histogram, sp_histogram, length_histogram, relative_k_histogram,
                                                                                                      configuration_bins, length_bins, k_bins)
                                                                                                      
            

    return path_histogram, sp_histogram, length_histogram, relative_k_histogram


@ray.remote
def run_aimless_TPS_fixed_length(total_trials : int, initial_path : np.ndarray, 
                                  path_length : int, equilibration_trials : int, 
                                  state_bounds : float , dt_D : float, configuration_bins : np.ndarray, 
                                  length_bins : np.ndarray, k_bins : np.ndarray, max_dk : int):
    
    return _run_aimless_TPS_fixed_length(total_trials, initial_path, 
                                         path_length, equilibration_trials, 
                                         state_bounds , dt_D, configuration_bins, 
                                         length_bins, k_bins, max_dk)



if __name__ == "__main__":

    ###
    # Parameters
    ###

    n_threads = 24
    total_trials = int(5e6) 
    equilibration_trials = 1000

    path_length = int(sys.argv[1])
    dt_D = 1e-2
    max_dk = 25

    state_bound_A = -5
    state_bound_B = 4

    configuration_bins = np.linspace(-9, 6, 200)
    length_bins = np.linspace(0, 10000, 200)
    k_bins = np.linspace(0, 1, 200)

    initial_path = np.linspace(state_bound_A - 0.01, state_bound_B + 0.01, path_length)

    ###
    # Run
    ###

    jobs = [run_aimless_TPS_fixed_length.remote(total_trials, initial_path, 
                                                path_length, equilibration_trials, 
                                                np.array([state_bound_A, state_bound_B ]) , dt_D, configuration_bins, 
                                                length_bins, k_bins, max_dk) for i in range(n_threads)]
    results = ray.get(jobs)

    trajectory_ensemble = np.sum([r[0] for r in results], axis=0)
    SP_ensemble = np.sum([r[1] for r in results], axis=0)

    np.save(f"data/fixed_length/aimless/path_density_{path_length}.npy", trajectory_ensemble)
    np.save(f"data/fixed_length/aimless/sp_density_{path_length}.npy", SP_ensemble)






