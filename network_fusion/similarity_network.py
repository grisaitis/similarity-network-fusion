from network_fusion.local_similarities import calculate_local_similarities
from network_fusion.weights import calculate_weights
from network_fusion.neighborhoods import calculate_neighborhoods
from network_fusion.distances import calculate_distances
from network_fusion.epsilon import calculate_epsilon


class SimilarityNetwork(object):
    def __init__(self, weights, similarities, k, mu):
        self.weights = weights
        self.similarities = similarities
        self.k = k
        self.mu = mu

    @classmethod
    def from_features(cls, data, k, mu):
        distances = calculate_distances(data)
        neighbor_indices, neighbor_distances = \
            calculate_neighborhoods(distances, k)
        epsilon = calculate_epsilon(distances, neighbor_distances)
        weights = calculate_weights(distances, epsilon, mu)
        similarities = calculate_local_similarities(weights, neighbor_indices)
        return cls(weights, similarities, k, mu)
