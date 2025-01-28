
from numba import njit
import numpy as np
import ray
import sys 

from scripts.system import force_function, check_reactive
from scripts.path_generation_flexible import shoot_two_way_flexible
from scripts.statistics import initialize_statistics, update_statistics


@njit
def _run_mullen2015_flexible_length(total_trials : int, initial_path : np.ndarray, equilibration_trials : int, 
                                  state_bounds : float , dt_D : float, configuration_bins : np.ndarray, 
                                  length_bins : np.ndarray, k_bins : np.ndarray, max_dk : int):
    
    current_path = initial_path.copy()
    
    # Get the index of the path center
    current_sp_index = current_path.shape[0] // 2
    
    # Statistics
    path_histogram, sp_histogram, length_histogram, relative_k_histogram = initialize_statistics(configuration_bins, length_bins, k_bins)

    current_indices = [current_sp_index, current_sp_index + max_dk]

    for trial in range(total_trials):

        sp_index = current_indices[np.random.randint(0, 2)]

        if sp_index >= 0 and sp_index < current_path.shape[0]:

            shooting_point = current_path[sp_index]
            
            # Generate forward and backward trajectory
            proposal_trj, proposal_index = shoot_two_way_flexible(shooting_point, state_bounds[0], state_bounds[1], dt_D, force_function)

            # If it is reactive, accept
            is_reactive_path = check_reactive(proposal_trj, state_bounds[0], state_bounds[1])
            if is_reactive_path:

                if np.random.random() < 0.5:
                    new_indices = [proposal_index, proposal_index + max_dk]
                else:
                    new_indices = [proposal_index - max_dk, proposal_index]

                new_index = new_indices[np.random.randint(0, 2)]

                if new_index >= 0 and new_index < proposal_trj.shape[0]:

                    current_path = proposal_trj
                    current_sp_index = new_index
                    current_indices = new_indices

        # Statistics
        if trial > equilibration_trials:
            path_histogram, sp_histogram, length_histogram, relative_k_histogram =  update_statistics(current_path, current_sp_index,
                                                                                                      path_histogram, sp_histogram, length_histogram, relative_k_histogram,
                                                                                                      configuration_bins, length_bins, k_bins)
                                                                                                      
            

    return path_histogram, sp_histogram, length_histogram, relative_k_histogram



@ray.remote
def run_mullen2015_flexible_length(total_trials : int, initial_path : np.ndarray, equilibration_trials : int, 
                                  state_bounds : float , dt_D : float, configuration_bins : np.ndarray, 
                                  length_bins : np.ndarray, k_bins : np.ndarray, max_dk : int):
    
    return _run_mullen2015_flexible_length(total_trials, initial_path, equilibration_trials, 
                                            state_bounds , dt_D, configuration_bins, 
                                            length_bins, k_bins, max_dk)



if __name__ == "__main__":

    ###
    # Parameters
    ###

    n_threads = 24
    total_trials = int(5e5) 
    equilibration_trials = 1000

    dt_D = 1e-2
    max_dk = 25

    state_bound_A = -5
    state_bound_B = 4

    configuration_bins = np.linspace(-9, 6, 200)
    length_bins = np.linspace(0, 2000, 201)
    k_bins = np.linspace(0, 1, 200)

    initial_path = np.linspace(state_bound_A - 0.01, state_bound_B + 0.01, 1500)

    ###
    # Run
    ###

    jobs = [run_mullen2015_flexible_length.remote(total_trials, initial_path, equilibration_trials, 
                                                  np.array([state_bound_A, state_bound_B ]) , dt_D, configuration_bins, 
                                                  length_bins, k_bins, max_dk) for i in range(n_threads)]
    results = ray.get(jobs)

    trajectory_ensemble = np.sum([r[0] for r in results], axis=0)
    SP_ensemble = np.sum([r[1] for r in results], axis=0)
    length_histogram = np.sum([r[2] for r in results], axis=0)
    k_histogram = np.sum([r[3] for r in results], axis=0)

    np.save(f"data/flexible_length/mullen_2015/path_density.npy", trajectory_ensemble)
    np.save(f"data/flexible_length/mullen_2015/sp_density.npy", SP_ensemble)
    np.save(f"data/flexible_length/mullen_2015/length_histogram.npy", length_histogram)
    np.save(f"data/flexible_length/mullen_2015/k_histogram.npy", k_histogram)






