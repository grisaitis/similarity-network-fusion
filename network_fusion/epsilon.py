import numpy as np


def calculate_epsilon(distances, neighbor_distances):
    mean_neighbor_distance = np.mean(neighbor_distances, axis=1)
    assert mean_neighbor_distance.shape == (distances.shape[0],)
    mean_i_plus_mean_j = \
        mean_neighbor_distance[None, :] + mean_neighbor_distance[:, None]
    epsilon = (mean_i_plus_mean_j + distances) / 3
    return epsilon
