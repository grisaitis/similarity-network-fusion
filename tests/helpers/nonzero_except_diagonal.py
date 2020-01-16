import numpy as np


def nonzero_except_diagonal(a):
    a_plus_identity = a + np.identity(a.shape[0], dtype=a.dtype)
    return a_plus_identity.all() and not a.all()
