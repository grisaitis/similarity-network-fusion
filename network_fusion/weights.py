def calculate_weights(distance, epsilon, mu):
  weights = np.exp(-distance * distance / mu / epsilon)
  return weights