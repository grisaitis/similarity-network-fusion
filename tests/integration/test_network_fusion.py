import pytest

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


@pytest.mark.skip("not implemented")
def test_fusion_step():
  # p_of_this_net
  # s_of_other_net
  # assert p_new[i, j] = s[i, j] * p[i, j] * s[j, i]
  pass


@pytest.mark.skip("WIP")
def test_similarity_network_fusion():
  '''
  networks = SimilarityNetwork(x_1), SimilarityNetwork(x_2)
  fuser = NetworkFuser()
  network_iterations = [networks]
  steps = 10
  for t in range(steps):
    networks_t = network_iterations[-1]
    networks_t_plus_one = fuser.fuse(networks_t)
    network_iterations.append(networks_t_plus_one)
  '''
  pass
