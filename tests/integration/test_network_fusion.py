import heapq

import numpy as np
import scipy.spatial as spatial


def calculate_distances(x: np.ndarray):
  """Calculates euclidean distances between each row of x
  
  Arguments:
      x {np.ndarray[np.float]} -- a matrix with n rows
  
  Returns:
      [type] -- a symmetric matrix of shape (n, n)
  """
  n = x.shape[0]
  distance = np.zeros(shape=(n, n), dtype=x.dtype)
  for i, x_i in enumerate(x):
    for j in range(i + 1, n):
      distance[i, j] = distance[j, i] = spatial.distance.euclidean(x_i, x[j])
  return distance


def calculate_neighborhoods(distance, k_neighbors):
  n = distance.shape[0]
  neighbor_indices = np.empty((n, k_neighbors), dtype=np.uint32)
  neighbor_distances = np.empty((n, k_neighbors), dtype=distance.dtype)
  for i, distance_i in enumerate(distance):
    distances_from_i_excluding_i = \
      [(distance_i_j, j) for j, distance_i_j in enumerate(distance_i) if i != j]
    distances_and_neighbors = heapq.nsmallest(k_neighbors, distances_from_i_excluding_i)
    neighbor_distances[i], neighbor_indices[i] = zip(*distances_and_neighbors)
  return neighbor_indices, neighbor_distances


def calculate_epsilon(distance, neighbor_distances):
  mean_neighbor_distance = np.mean(neighbor_distances, axis=1)
  epsilon = np.ones_like(distance)
  for i, distance_i in enumerate(distance):
    for j, distance_i_j in enumerate(distance_i[i + 1:]):
      epsilon[i, j] = epsilon[j, i] = (mean_neighbor_distance[i] + mean_neighbor_distance[j] + distance_i_j) / 3
  return epsilon


def calculate_normalized_weights(weights):
  raise NotImplementedError


def calculate_local_similarities(weights, neighbor_indices):
  n = weights.shape[0]
  neighbor_weight_sums = np.zeros(shape=(n,), dtype=weights.dtype)
  # to do: calculate and store neighbor_weight_sums
  inverse_neighborhood_weights = np.zeros_like(weights)
  for i, weights_i in enumerate(weights):
    # add up weights[k_neighbors] for k_neighbors in neighborhood
    for neighbor_index in neighbor_indices[i]:
      inverse_neighborhood_weights[i, neighbor_index] = 1.0 / neighbor_weight_sums[i]
  similarity = weights * inverse_neighborhood_weights


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
  weights = np.exp(-distance * distance / mu / epsilon)
  assert is_symmetric_matrix(weights)
  normalized_weights = calculate_normalized_weights(weights)
  assert is_symmetric_matrix(normalized_weights)
  local_similarities = calculate_local_similarities(weights, neighbor_indices)
