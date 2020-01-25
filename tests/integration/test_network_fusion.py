import numpy as np
import pytest

from network_fusion.distances import calculate_distances
from network_fusion.epsilon import calculate_epsilon
from network_fusion.local_similarities import calculate_local_similarities
from network_fusion.neighborhoods import calculate_neighborhoods
from network_fusion.normalized_weights import calculate_normalized_weights
from network_fusion.weights import calculate_weights
from tests.helpers.fake_data import make_fake_data


def test_similarity_network_fusion():
    rs = np.random.RandomState(0)
    n_datasets = 3
    feature_matrices = [
        make_fake_data(n=20, m=3, random_state=rs)
        for _ in range(n_datasets)]
    mu = 0.5
    k = 5
    distance_graphs = [calculate_distances(f) for f in feature_matrices]
    neighborhoods = [calculate_neighborhoods(d, k) for d in distance_graphs]
    epsilons = [calculate_epsilon(d, n[1])
                for d, n in zip(distance_graphs, neighborhoods)]
    weight_graphs = [calculate_weights(d, e, mu)
                     for d, e in zip(distance_graphs, epsilons)]
    normalized_weight_graphs = [calculate_normalized_weights(w)
                                for w in weight_graphs]
    local_similarity_graphs = [
        calculate_local_similarities(nw, n[0])
        for nw, n in zip(normalized_weight_graphs, neighborhoods)]
    T_iterations = 1
    weights_by_iteration = [normalized_weight_graphs]
    for t in range(T_iterations):
        weights_at_iteration_t_plus_one = list(None for _ in range(n_datasets))
        for v, s_v in enumerate(local_similarity_graphs):
            other_normalized_weights = list(weights_by_iteration[t])
            other_normalized_weights.pop(v)
            mean_other_normalized_weights = \
                np.mean(other_normalized_weights, axis=0)
            assert mean_other_normalized_weights.shape == s_v.shape
            weights_at_iteration_t_plus_one[v] = \
                s_v @ mean_other_normalized_weights @ s_v.T
            assert weights_at_iteration_t_plus_one[v].shape \
                == weight_graphs[v].shape
        weights_by_iteration.append(weights_at_iteration_t_plus_one)
