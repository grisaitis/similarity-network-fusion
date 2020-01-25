import heapq

import numpy as np


def calculate_neighborhoods(distances, k_neighbors):
    """Calculate neighbor distances and indices of neighbors for each instance

    Arguments:
        distances {numpy.ndarray} -- pairwise distance matrix (symmetric)
        k_neighbors {int} -- size to use for neighborhoods

    Returns:
        (numpy.ndarray, numpy.ndarray) -- neighbor indices, neighbor distances
    """
    n = distances.shape[0]
    neighbor_indices = np.empty((n, k_neighbors), dtype=np.uint32)
    neighbor_distances = np.empty((n, k_neighbors), dtype=distances.dtype)
    for i, distances_i in enumerate(distances):
        distances_from_i_excluding_i = [
            (distances_i_j, j)
            for j, distances_i_j in enumerate(distances_i) if i != j]
        distances_and_neighbors = heapq.nsmallest(
            k_neighbors, distances_from_i_excluding_i)
        neighbor_distances[i], neighbor_indices[i] = \
            zip(*distances_and_neighbors)
    return neighbor_indices, neighbor_distances
