import heapq

import numpy as np
import scipy.spatial as spatial

def calculate_distances(x: np.ndarray[np.float]):
  """Calculates euclidean distances between each row of x
  
  Arguments:
      x {np.ndarray[np.float]} -- a matrix with n rows
  
  Returns:
      [type] -- a symmetric matrix of shape (n, n)
  """
  n = x.shape[0]
  distance = np.zeros(shape=(n, n), dtype=x.dtype)
  for i, x_i in enumerate(x):
    for j, x_j in enumerate(x[i + 1:]):
      distance[i, j] = distance[j, i] = spatial.distance.euclidean(x_i, x_j)
  return distance

def test_makes_fused_network():
  rs = np.random.RandomState(0)
  n, p = 20, 10
  x = random_state.uniform(size=n * p).reshape((n, p))
  mu = 0.5
  k_neighbors = 5

  distance = calculate_distances(x)

  neighbors = dict()
  neighbor_distances = np.array((n, k), )
  for i, distance_i in enumerate(distance):
    distances_from_i_excluding_i = [(distance_i_j, j) for j, distance_i_j in distance_i if i != j]
    distances_and_neighbors = heapq.nsmallest(k_neighbors, distances_from_i_excluding_i)
    neighbor_distances[i], neighbors[i] = zip(*distances_and_neighbors)

  epsilon = np.zeros_like(distance)
  for i, distance_i in enumerate(distance):
    epsilon[i, i] = 2 * mean_distance_to_nearest[i] / 3
    for j, distance_i_j in enumerate(distance_i[i + 1:]):
      epsilon[i, j] = epsilon[j, i] = (mean_distance_to_nearest[i] \
        + mean_distance_to_nearest[j] + distance_i_j) / 3
  weights = np.exp(-distance * distance / mu / epsilon)

  normalized_weights = compute_normalized_weights(weights)

  sparse_similarity = compute_sparse_similarity(weights, neighbors)

  inverse_neighborhood_weights = np.zeros_like(weights)
  for i, distance_i in enumerate(distance):
    neighbor_tuples = \
      [(distance_i_j, j) for j, distance_i_j in distance_i if i != j]
    # make neighborhood with nsmallest
    k_smallest_tuples = heapq.nsmallest(k_neighbors, neighbor_tuples)
    # add up weights[k] for k in neighborhood
  '''
  '''
  similarity = weights * inverse_neighborhood_weights
  # done with similarity
  return distance, weights, similarity
  # similarity_net_1 = SimilarityNetwork(X1)
  # fuser = NetworkFuser()
  # fused_network = fuser.fuse(networks)
