import jax.numpy as np

from network_fusion.similarity_kernel import SimilarityKernel


def test_similarity_kernel():
  kernel = SimilarityKernel(mu=1.0)
  x = np.array([0])
  assert kernel(x, x) == 1
  x = np.array([3])
  assert kernel(x, x) == 1
  x, y = np.array([0]), np.array([5])
  assert kernel(x, y) == np.exp()
