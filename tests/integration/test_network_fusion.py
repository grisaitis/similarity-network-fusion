import heapq

import numpy as np

from network_fusion.distances import calculate_distances
from network_fusion.local_similarities import calculate_local_similarities
from network_fusion.neighborhoods import calculate_neighborhoods
from network_fusion.epsilon import calculate_epsilon
from network_fusion.normalized_weights import calculate_normalized_weights
from network_fusion.weights import calculate_weights


from tests.helpers import nonzero_except_diagonal


def test_makes_fused_network():
  random_state = np.random.RandomState(0)
  n, p = 5, 4
  x = random_state.uniform(size=n * p).reshape((n, p))
  mu = 0.5
  k_neighbors = 3
  distance = calculate_distances(x)
  assert np.allclose(distance, distance.T), "must be symmetric"
  assert nonzero_except_diagonal(distance)
  neighbor_indices, neighbor_distances = calculate_neighborhoods(distance, k_neighbors)
  epsilon = calculate_epsilon(distance, neighbor_distances)
  assert np.allclose(epsilon, epsilon.T), "must be symmetric"
  weights = calculate_weights(distance, epsilon, mu)
  assert np.allclose(weights, weights.T), "must be symmetric"
  normalized_weights = calculate_normalized_weights(weights)
  assert np.allclose(normalized_weights, normalized_weights.T), "must be symmetric"
  local_similarities = calculate_local_similarities(weights, neighbor_indices)


