
from numba import njit
import numpy as np
import ray
import sys 

from scripts.system import force_function
from scripts.integrator import update_positions


@njit
def _run_eq(n_steps : float, state_bounds : np.ndarray, dt_D : float, configuration_bins : np.ndarray, length_bins : np.ndarray):

    prefactor = np.sqrt(2*dt_D)
    current_x = state_bounds[0] - 0.01

    last_state = -1
    current_TP_length = 0
    current_configuration_histogram = np.zeros(configuration_bins.shape[0] - 1)

    configuration_histogram = np.zeros((2, configuration_bins.shape[0] - 1))
    length_histogram = np.zeros(length_bins.shape[0] - 1)

    for step in range(n_steps):

        current_force = force_function(current_x)
        current_x = update_positions(current_x, current_force, dt_D, prefactor)

        state_A = current_x < state_bounds[0]
        state_B = current_x > state_bounds[1]

        if not state_A and not state_B: 

            current_TP_length += 1
            current_configuration_histogram += np.histogram(current_x, bins=configuration_bins)[0]
        
        elif state_A and last_state == 1:   

            last_state = -1
            
            # add 2 since we missed the points in the stable state
            length_histogram += np.histogram(current_TP_length + 2, bins=length_bins)[0]

            current_configuration_histogram += np.histogram(current_x, bins=configuration_bins)[0]
            configuration_histogram[0] += current_configuration_histogram
            configuration_histogram[1] += 1/(current_TP_length + 2) * current_configuration_histogram

            current_TP_length = 0
            current_configuration_histogram.fill(0)

        elif state_B and last_state == -1:

            last_state = 1
            
            # add 2 since we missed the points in the stable state
            length_histogram += np.histogram(current_TP_length + 2, bins=length_bins)[0]

            current_configuration_histogram += np.histogram(current_x, bins=configuration_bins)[0]
            configuration_histogram[0] += current_configuration_histogram
            configuration_histogram[1] += 1/(current_TP_length + 2) * current_configuration_histogram
            
            current_TP_length = 0
            current_configuration_histogram.fill(0)
        
        else:
            current_TP_length = 0

            current_configuration_histogram.fill(0)
            current_configuration_histogram += np.histogram(current_x, bins=configuration_bins)[0]


    return configuration_histogram, length_histogram


@ray.remote
def run_eq(n_steps : float, state_bounds : np.ndarray, dt_D : float, configuration_bins : np.ndarray, length_bins : np.ndarray):
    return  _run_eq(n_steps, state_bounds, dt_D, configuration_bins, length_bins)



if __name__ == "__main__":

    ###
    # Parameters
    ###

    n_threads = 24
    n_steps = int(1e9) 

    dt_D = 1e-2

    state_bound_A = -5
    state_bound_B = 4

    configuration_bins = np.linspace(-9, 6, 200)
    length_bins = np.linspace(0, 2000, 201)

    ###
    # Run
    ###

    jobs = [run_eq.remote(n_steps, np.array([state_bound_A, state_bound_B]) , dt_D, configuration_bins, length_bins) for i in range(n_threads)]
    results = ray.get(jobs)

    eq_path_density = np.sum([r[0] for r in results], axis=0)
    length_histogram = np.sum([r[1] for r in results], axis=0)

    np.save(f"data/flexible_length/equilibrium/eq_path_density.npy", eq_path_density)
    np.save(f"data/flexible_length/equilibrium/length_histogram.npy", length_histogram)






