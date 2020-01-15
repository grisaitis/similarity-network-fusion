import numpy as np


def calculate_epsilon(distance, neighbor_distances):
  mean_neighbor_distance = np.mean(neighbor_distances, axis=1)
  epsilon = np.ones_like(distance)
  for i, distance_i in enumerate(distance):
    for j, distance_i_j in enumerate(distance_i[i + 1:], start=i + 1):
      epsilon[i, j] = epsilon[j, i] = (mean_neighbor_distance[i] + mean_neighbor_distance[j] + distance_i_j) / 3
  return epsilon
