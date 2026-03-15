import numpy as np

from source.kernels.kernel import Kernel


# k(x,y) = variance * exp(-||x-y||^2 / (2 * lengthscale^2))
class RBFKernel(Kernel):
    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        if lengthscale <= 0:
            raise ValueError("lengthscale must be positive")
        if variance <= 0:
            raise ValueError("variance must be positive")
        self.lengthscale = float(lengthscale)
        self.variance = float(variance)

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        # Pre-scale inputs so squared_euclidean gives ||x-y||^2 / ell^2
        dists2 = self._squared_euclidean(X / self.lengthscale, Y / self.lengthscale)
        return self.variance * np.exp(-0.5 * dists2)

    def diag(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self.variance)
