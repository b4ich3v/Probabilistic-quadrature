import numpy as np
from numpy.typing import ArrayLike
from typing import Optional
from scipy.linalg import cho_factor, cho_solve

from source.kernels.kernel import Kernel
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.utils import ensure_2d


class GaussianProcess:
    def __init__(self, kernel: Kernel, noise: float = 0.0, jitter: float = 0.0):
        if noise < 0:
            raise ValueError("noise must be non-negative")
        if jitter < 0:
            raise ValueError("jitter must be non-negative")
        self.kernel = kernel
        self.noise = float(noise)
        self.jitter = float(jitter)
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.L_factor: Optional[tuple] = None

    def _check_fitted(self) -> None:
        if self.X_train is None or self.y_train is None or self.L_factor is None:
            raise RuntimeError("GaussianProcess is not fitted yet")

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GaussianProcess":
        X = ensure_2d(X)
        y = np.asarray(y, dtype=float)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have same first dimension")

        K = self.kernel(X, X)
        if self.noise > 0.0:
            K = K + (self.noise ** 2) * np.eye(len(X))
        if self.jitter > 0.0:
            K = K + self.jitter * np.eye(len(X))

        try:
            L_factor = cho_factor(K)
        except np.linalg.LinAlgError:
            raise ValueError("Kernel matrix is not positive definite. Try increasing jitter or noise.")

        self.X_train = X
        self.y_train = y
        self.L_factor = L_factor
        return self

    def solve(self, v: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return cho_solve(self.L_factor, v)

    def predict(self, X_test: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        X_test = ensure_2d(X_test)
        K_s = self.kernel(self.X_train, X_test)
        alpha = cho_solve(self.L_factor, self.y_train)
        mean = K_s.T @ alpha
        v = cho_solve(self.L_factor, K_s)
        var = self.kernel.diag(X_test) - np.sum(K_s * v, axis=0)
        return mean, np.maximum(var, 0.0)