import numpy as np


from network_fusion.normalized_weights import calculate_normalized_weights


def test_normalized_weights():
    weights = np.array([
        [1, 2, 6],
        [2, 1, 10],
        [6, 10, 1]
    ])
    weights_normed = calculate_normalized_weights(weights)
    assert weights_normed[0, 0] == weights_normed[1, 1] \
        == weights_normed[2, 2] == 0.5
    assert weights_normed[0, 1] == \
        weights[0, 1] / (2 * (weights[0, 1] + weights[0, 2]))
    assert weights_normed[1, 0] == \
        weights[1, 0] / (2 * (weights[1, 0] + weights[1, 2]))
    assert weights_normed[1, 2] == \
        weights[1, 2] / (2 * (weights[1, 0] + weights[1, 2]))
    assert weights_normed[0, 2] == \
        weights[0, 2] / (2 * (weights[0, 1] + weights[0, 2]))
