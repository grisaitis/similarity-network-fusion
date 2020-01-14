import heapq

import numpy as np

from network_fusion.distances import calculate_distances
from network_fusion.local_similarities import calculate_local_similarities
from network_fusion.neighborhoods import calculate_neighborhoods
from network_fusion.epsilon import calculate_epsilon
from network_fusion.normalized_weights import calculate_normalized_weights
from network_fusion.weights import calculate_weights


def is_symmetric_matrix(a):
  return np.rank(a) == 2 \
    and a.shape[0] == a.shape[1] \
    and np.allclose(a, a.T)


def nonzero_except_diagonal(a):
  a_plus_identity = a + np.identity(a.shape[0], dtype=a.dtype)
  return a_plus_identity.all() and not a.all()


def test_makes_fused_network():
  random_state = np.random.RandomState(0)
  n, p = 5, 4
  x = random_state.uniform(size=n * p).reshape((n, p))
  mu = 0.5
  k_neighbors = 3
  distance = calculate_distances(x)
  assert is_symmetric_matrix(distance)
  assert nonzero_except_diagonal(distance)
  neighbor_indices, neighbor_distances = calculate_neighborhoods(distance, k_neighbors)
  epsilon = calculate_epsilon(distance, neighbor_distances)
  assert is_symmetric_matrix(epsilon)
  weights = calculate_weights(distance, epsilon, mu)
  assert is_symmetric_matrix(weights)
  normalized_weights = calculate_normalized_weights(weights)
  assert is_symmetric_matrix(normalized_weights)
  local_similarities = calculate_local_similarities(weights, neighbor_indices)


