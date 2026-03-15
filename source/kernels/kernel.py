from abc import ABC, abstractmethod
import numpy as np


# Abstract positive-definite kernel k(X, Y) for Gaussian processes
class Kernel(ABC):
    # ||x_i - y_j||^2 via the expansion ||x||^2 + ||y||^2 - 2<x,y>, clamped >= 0
    def _squared_euclidean(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        XX = np.sum(np.square(X), axis=1)[:, None]
        YY = np.sum(np.square(Y), axis=1)[None, :]
        cross = X @ Y.T
        return np.maximum(XX + YY - 2.0 * cross, 0.0)

    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def diag(self, X: np.ndarray) -> np.ndarray: ...