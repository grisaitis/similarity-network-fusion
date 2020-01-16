import numpy as np


def calculate_weights(distances, epsilon, mu):
    return np.exp(-distances * distances / (mu * epsilon))
