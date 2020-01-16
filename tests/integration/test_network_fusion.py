import pytest


@pytest.mark.skip("not implemented")
def test_fusion_step():
    # p_of_this_net
    # s_of_other_net
    # assert p_new[i, j] = s[i, j] * p[i, j] * s[j, i]
    pass


@pytest.mark.skip("WIP")
def test_similarity_network_fusion():
    '''
    networks = SimilarityNetwork(x_1), SimilarityNetwork(x_2)
    fuser = NetworkFuser()
    network_iterations = [networks]
    steps = 10
    for t in range(steps):
        networks_t = network_iterations[-1]
        networks_t_plus_one = fuser.fuse(networks_t)
        network_iterations.append(networks_t_plus_one)
    '''
    pass
