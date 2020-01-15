import numpy as np

from network_fusion.distances import calculate_distances
from network_fusion.local_similarities import calculate_local_similarities
from network_fusion.neighborhoods import calculate_neighborhoods
from network_fusion.epsilon import calculate_epsilon
from network_fusion.normalized_weights import calculate_normalized_weights
from network_fusion.weights import calculate_weights


def test_local_similarities():
  random_state = np.random.RandomState(0)
  n, p = 5, 4
  x = random_state.uniform(size=n * p).reshape((n, p))
  mu = 0.5
  k_neighbors = 3

  network = compute_network_from_features(x, mu, k_neighbors)

  assert np.allclose(
    network.distances,
    

  
  assert similarity_network.
  distances = calculate_distances(x)
  neighbor_indices, neighbor_distances = \
    calculate_neighborhoods(distances, k_neighbors)
  epsilon = calculate_epsilon(distances, neighbor_distances)
  weights = calculate_weights(distances, epsilon, mu)
  diagonals_equal_1 = all(weights[i, i] == 1 for i in range(weights.shape[0]))
  assert diagonals_equal_1
  normalized_weights = calculate_normalized_weights(weights)
  assert np.sum(normalized_weights, axis=0) == 1
  assert np.sum(normalized_weights, axis=1) == 1
  # to do: assert that diagonal values are 0.5
  local_similarities = calculate_local_similarities(weights, neighbor_indices)
  rows_are_sparse = np.sum(local_similarities != 0, axis=0) == k_neighbors
  assert rows_are_sparse
  columns_are_sparse = np.sum(local_similarities != 0, axis=0) == k_neighbors
  assert columns_are_sparse
  # to do: assert that local similarities are not symmetric
  return normalized_weights, local_similarities
