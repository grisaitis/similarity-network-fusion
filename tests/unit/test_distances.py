import numpy as np

from network_fusion.distances import calculate_distances

from tests.helpers import nonzero_except_diagonal


def test_calculates_the_right_distance_n_2():
  x = np.array([[3], [5]])
  distances = calculate_distances(x)
  assert np.allclose(distances, np.array([
    [0, 2],
    [2, 0]
  ]))


def test_calculates_the_right_distance_n_4():
  x = np.array([[3], [3], [4], [6]])
  distances = calculate_distances(x)
  assert np.allclose(distances, np.array([
    [0, 0, 1, 3],
    [0, 0, 1, 3],
    [1, 1, 0, 2],
    [3, 3, 2, 0]
  ]))


def test_is_shaped_correctly():
  random_state = np.random.RandomState(0)
  n, p = 100, 40
  x = random_state.uniform(size=n * p).reshape((n, p))
  distances = calculate_distances(x)
  assert distances.shape[0] == x.shape[0]
  assert distances.shape[1] == x.shape[0]
  assert distances.shape[0] == distances.shape[1], "must be square"
  assert np.allclose(distances, distances.T), "must be symmetric"
  assert nonzero_except_diagonal(distances)
