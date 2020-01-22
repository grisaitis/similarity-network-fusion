import numpy as np
import pytest

from network_fusion.weights import calculate_weights
from network_fusion.epsilon import calculate_epsilon
from network_fusion.neighborhoods import calculate_neighborhoods


def test_is_correct_value():
    mu = 0.5
    d = np.array([
        [0, 5],
        [5, 0]])
    e = np.array([
        [(5 + 5 + 0) / 3, (5 + 5 + 5) / 3],
        [(5 + 5 + 5) / 3, (5 + 5 + 0) / 3]
    ])
    w = calculate_weights(d, e, mu)
    assert w[0, 0] == 1  # because distance is zero
    assert w[1, 1] == 1
    assert w[0, 1] == np.exp(-d[0, 1] ** 2 / (mu * e[0, 1]))
    assert w[1, 0] == np.exp(-d[1, 0] ** 2 / (mu * e[1, 0]))


def test_is_correct_value_bigger():
    mu = 0.5
    k = 2
    d = np.array([
        [0, 5, 9, 3],
        [5, 0, 7, 2],
        [9, 7, 0, 1],
        [3, 2, 1, 0]])
    n_indices, n_distances = calculate_neighborhoods(d, k)
    e = calculate_epsilon(d, n_distances)
    w = calculate_weights(d, e, mu)
    assert all(w[i, i] == 1 for i in range(d.shape[0]))
    assert w[0, 2] == np.exp(-9 ** 2 / (mu * ((4 + 4 + 9) / 3)))
    assert w[3, 2] == np.exp(-1 ** 2 / (mu * ((1.5 + 4 + 1) / 3)))


@pytest.mark.skip("WIP - changing input data")
def test_mu_changes_weights():
    rs = np.random.RandomState(0)
    distances = rs.uniform(size=100).reshape((10, 10))
    mu_1, mu_2 = 0.4, 0.6
    k = 1
    neighbor_distances = np.array([2, 2, 3])
    neighbor_indices = np.array([[1], [0], [0]])
    epsilon = calculate_epsilon(distances, neighbor_distances)
    weights_1 = calculate_weights(distances, epsilon, mu_1)
    weights_2 = calculate_weights(distances, epsilon, mu_2)
    assert weights.shape == (n_samples, n_samples)
    assert weights.shape == (n_samples, n_samples)
    assert weights_1[0, 1] == np.exp(2 / mu_1 / 1)
