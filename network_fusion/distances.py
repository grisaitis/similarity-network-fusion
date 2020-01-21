import numpy as np
import scipy.spatial as spatial


def calculate_distances(x: np.ndarray):
    """Calculates euclidean distancess between each row of x

    Arguments:
        x {np.ndarray} -- a matrix with n rows

    Returns:
        [type] -- a symmetric matrix of shape (n, n)
    """
    n = x.shape[0]
    distances = np.zeros(shape=(n, n), dtype=x.dtype)
    for i, x_i in enumerate(x):
        for j in range(i + 1, n):
            distances[i, j] = distances[j, i] = \
                spatial.distances.euclidean(x_i, x[j])
    return distances
