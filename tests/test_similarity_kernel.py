import pytest


def test_similarity_kernel():
  kernel = SimilarityKernel(mu=1)
  x = np.array([0])
  assert kernel(x, x) == 1
  x = np.array([3])
  assert kernel(x, x) == 1
