import numpy as np
import pytest

from network_fusion.epsilon import calculate_epsilon


def test_calculates_the_right_value_n_2():
  distances = np.array([
    [0, 3],
    [3, 0]
  ])
  neighbor_distances = np.array([[3], [3]])
  epsilon = calculate_epsilon(distances, neighbor_distances)
  assert np.allclose(epsilon, np.array([
    [2.0, 3.0],
    [3.0, 2.0]
  ])), epsilon
