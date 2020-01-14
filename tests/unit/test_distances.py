import numpy as np

from network_fusion.distances import calculate_distances

from tests.helpers import nonzero_except_diagonal


def test_calculates_the_right_distance():
  x = np.array([[3], [3], [4], [6]])
  distances = calculate_distances(x)
  assert distances[0, 0] == 0
  assert distances[0, 1] == 0
  assert distances[0, 2] == 1
  assert distances[0, 3] == 3
  assert distances[1, 2] == 1
  assert distances[1, 3] == 3
  assert distances[2, 3] == 2


def test_is_shaped_correctly():
  random_state = np.random.RandomState(0)
  x = np.array([[1], [2], [4]])
  distances = calculate_distances(x)
  assert distances.shape[0] == x.shape[0]
  assert distances.shape[1] == x.shape[0]
  assert distances.shape[0] == distances.shape[1], "must be square"
  assert np.allclose(distances, distances.T), "must be symmetric"
  assert nonzero_except_diagonal(distances)

def test_is_good_with_bigger_data():
  random_state = np.random.RandomState(0)
  n, p = 100, 40
  x = random_state.uniform(size=n * p).reshape((n, p))
  distances = calculate_distances(x)
  assert np.allclose(distances, distances.T), "must be symmetric"
  assert nonzero_except_diagonal(distances)
