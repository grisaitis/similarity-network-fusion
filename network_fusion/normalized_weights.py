import numpy as np


def calculate_normalized_weights(weights):
    row_sums_without_diagonals = np.sum(weights, axis=1) - weights.diagonal()
    row_sums_without_diagonals = row_sums_without_diagonals.reshape(-1, 1)
    assert row_sums_without_diagonals.shape == (weights.shape[0], 1)
    normalized_weights = weights / (2 * row_sums_without_diagonals)
    for i in range(normalized_weights.shape[0]):
        normalized_weights[i, i] = 0.5
    return normalized_weights
