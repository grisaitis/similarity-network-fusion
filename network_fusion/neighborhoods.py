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