
import numpy as np
from numba import njit

@njit 
def initialize_statistics(configuration_bins, length_bins, k_bins):

    path_histogram = np.zeros(configuration_bins.shape[0] - 1)
    sp_histogram = np.zeros(configuration_bins.shape[0] - 1)
    length_histogram = np.zeros(length_bins.shape[0] - 1)
    relative_k_histogram = np.zeros(k_bins.shape[0] - 1)

    return path_histogram, sp_histogram, length_histogram, relative_k_histogram


@njit
def update_statistics(current_path, current_sp_index, 
                       path_histogram, sp_histogram, length_histogram, relative_k_histogram, 
                       configuration_bins, length_bins, k_bins):
    
    path_histogram += np.histogram(current_path, configuration_bins)[0]
    sp_histogram += np.histogram(current_path[current_sp_index], configuration_bins)[0]

    length_histogram += np.histogram(len(current_path), length_bins)[0]
    relative_k_histogram += np.histogram(current_sp_index / len(current_path), k_bins)[0]

    return path_histogram, sp_histogram, length_histogram, relative_k_histogram