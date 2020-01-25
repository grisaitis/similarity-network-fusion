import numpy as np


def make_fake_data(n=30, m=100, random_state=None):
    """Make a nd.array with gaussian noise of shape (n, m)

    Keyword Arguments:
        n {int} -- number of rows (default: {30})
        m {int} -- number of columns or features (default: {100})
        random_state {np.random.RandomState} -- RandomState (default: {None})

    Returns:
        np.ndarray -- a numpy.ndarray with shape (n, m)
    """
    random_state = random_state or np.random.RandomState(0)
    return random_state.normal(size=n * m).reshape((n, m))
