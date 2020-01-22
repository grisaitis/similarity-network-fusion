import numpy as np


def make_fake_data(n=30, m=100, random_state=None):
    random_state = random_state or np.random.RandomState(0)
    return random_state.normal(size=n * m).reshape((n, m))
