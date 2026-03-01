import numpy as np

from source.kernels.kernel import Kernel


class Matern32Kernel(Kernel):
    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        self.lengthscale = float(lengthscale)
        self.variance = float(variance)

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        dists2 = self._squared_euclidean(X / self.lengthscale, Y / self.lengthscale)
        dists = np.sqrt(np.maximum(dists2, 0.0))
        scaled = np.sqrt(3.0) * dists
        return self.variance * (1.0 + scaled) * np.exp(-scaled)

    def diag(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self.variance)

