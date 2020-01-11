import jax.numpy as np


class SimilarityKernel(object):
  def __init__(self, mu):
    self.mu = mu

  def __call__(self, x_i, x_j, neighbors):
    squared_distance = np.sum(np.square(x_i - x_j))
    sqrt(4)
    
    # i_neighbor_distances = 
    epsilon = np.mean(i_neighbor_distances) + np.mean(j_neighbor_distances) + i_j_distance
    return np.exp(-squared_distance / (self.mu * epsilon))
