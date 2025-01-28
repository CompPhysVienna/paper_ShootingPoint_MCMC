
from numba import njit
import numpy as np
import ray
import sys 

from scripts.system import force_function
from scripts.integrator import update_positions


@njit
def _run_eq(n_steps : float, path_length : int, state_bounds : np.ndarray, dt_D : float, configuration_bins : np.ndarray, max_dk : int):

    prefactor = np.sqrt(2*dt_D)
    current_x = state_bounds[0] - 0.01

    configuration_buffer = np.zeros(path_length)
    current_conf_pointer = 0
    
    configuration_histogram = np.zeros(configuration_bins.shape[0] - 1)
    center_histogram = np.zeros(configuration_bins.shape[0] - 1)

    for step in range(n_steps):

        current_force = force_function(current_x)
        current_x = update_positions(current_x, current_force, dt_D, prefactor)

        configuration_buffer[current_conf_pointer] = current_x

        start_in_A = configuration_buffer[(current_conf_pointer + 1) % path_length] < state_bounds[0]
        start_in_B = configuration_buffer[(current_conf_pointer + 1) % path_length] > state_bounds[1]

        end_in_A = configuration_buffer[current_conf_pointer] < state_bounds[0]
        end_in_B = configuration_buffer[current_conf_pointer] > state_bounds[1]

        is_reactive_path = (start_in_A and end_in_B) or (start_in_B and end_in_A)

        if is_reactive_path: 

            configuration_histogram += np.histogram(configuration_buffer, bins=configuration_bins)[0]

            # center_histogram += np.histogram(configuration_buffer[(current_conf_pointer + path_length//2) % path_length], bins=configuration_bins)[0]
            center_histogram += np.histogram(configuration_buffer[(current_conf_pointer + path_length//2 - max_dk) % path_length], bins=configuration_bins)[0]
            center_histogram += np.histogram(configuration_buffer[(current_conf_pointer + path_length//2 + max_dk) % path_length], bins=configuration_bins)[0]

        current_conf_pointer += 1
        current_conf_pointer = current_conf_pointer % path_length

    return configuration_histogram, center_histogram


@ray.remote
def run_eq(n_steps : float, path_length : int, state_bounds : np.ndarray, dt_D : float, configuration_bins : np.ndarray, max_dk : int):
    return  _run_eq(n_steps, path_length, state_bounds, dt_D, configuration_bins, max_dk)



if __name__ == "__main__":

    ###
    # Parameters
    ###

    n_threads = 24
    n_steps = int(1e8) 

    path_length = int(sys.argv[1])
    dt_D = 1e-2
    max_dk = 25

    state_bound_A = -5
    state_bound_B = 4

    configuration_bins = np.linspace(-9, 6, 200)

    ###
    # Run
    ###

    jobs = [run_eq.remote(n_steps, path_length, np.array([state_bound_A, state_bound_B]) , dt_D, configuration_bins, max_dk) for i in range(n_threads)]
    results = ray.get(jobs)

    eq_path_density = np.sum([r[0] for r in results], axis=0)
    center_histogram = np.sum([r[1] for r in results], axis=0)

    np.save(f"data/fixed_length/equilibrium/eq_path_density_{path_length}.npy", eq_path_density)
    np.save(f"data/fixed_length/equilibrium/center_histogram_{path_length}.npy", center_histogram)






