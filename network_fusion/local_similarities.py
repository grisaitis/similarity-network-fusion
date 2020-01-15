import numpy as np


def calculate_local_similarities(weights, neighbor_indices):
  n = weights.shape[0]
  neighbor_weight_sums = np.zeros(shape=(n,), dtype=weights.dtype)
  # to do: calculate and store neighbor_weight_sums
  inverse_neighborhood_weights = np.zeros_like(weights)
  for i, weights_i in enumerate(weights):
    # add up weights[k_neighbors] for k_neighbors in neighborhood
    for neighbor_index in neighbor_indices[i]:
      inverse_neighborhood_weights[i, neighbor_index] = \
        1.0 / neighbor_weight_sums[i]
  return weights * inverse_neighborhood_weights
