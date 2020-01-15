import numpy as np
import pytest

from network_fusion.epsilon import calculate_epsilon


@pytest.mark.skip("will do this later")
def test_calculates_the_right_value_n_2():
  distances = np.array([
    [0, 3],
    [3, 0]
  ])
  neighbor_distances = np.array([[3], [3]])
  epsilon = calculate_epsilon(distances, neighbor_distances)
  assert np.allclose(epsilon, np.array([
    [1, ]
  ]))


def test_diagonals_are_one():
  distances = np.array([
    [0, 0.1, 1, 3],
    [0.1, 0, 1, 3],
    [1, 1, 0, 2],
    [3, 3, 2, 0]
  ])
  neighbor_distances = np.array([[0.1], [0.1], [1], [2]])
  epsilon = calculate_epsilon(distances, neighbor_distances)
  for i, row_i in enumerate(epsilon):
    assert row_i[i] == 1


@pytest.mark.skip("will do this later")
def test_is_square_symmetric():
  epsilon = None
  assert epsilon.shape[0] == epsilon.shape[1], "must be square"
  assert np.allclose(epsilon, epsilon.T), "must be symmetric"
