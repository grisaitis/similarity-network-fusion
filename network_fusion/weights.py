import numpy as np


def calculate_weights(distances, epsilon, mu):
    """Calculate weight matrix from distances and mu
    
    Arguments:
        distances {[type]} -- [description]
        epsilon {[type]} -- [description]
        mu {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """    
    return np.exp(-distances * distances / (mu * epsilon))
