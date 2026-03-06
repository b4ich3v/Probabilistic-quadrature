import numpy as np
from numpy.typing import ArrayLike
from typing import Optional

from source.kernels.kernel import Kernel
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.utils import (
    kernel_mean_vector, kernel_integral_variance, ensure_2d
)


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
        self.K: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None

    def _check_fitted(self) -> None:
        if self.X_train is None or self.y_train is None or self.K_inv is None:
            raise RuntimeError("GaussianProcess is not fitted yet")

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GaussianProcess":
        X = ensure_2d(X)
        y = np.asarray(y, dtype=float)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have same first dimension")

        K = self.kernel(X, X)
        if self.noise > 0.0:
            K = K + (self.noise**2) * np.eye(len(X))
        if self.jitter > 0.0:
            K = K + self.jitter * np.eye(len(X))

        K_inv = np.linalg.solve(K, np.eye(K.shape[0]))
        self.X_train = X
        self.y_train = y
        self.K = K
        self.K_inv = K_inv
        return self

    def predict(self, X_test: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        X_test = ensure_2d(X_test)
        X_train = self.X_train
        y_train = self.y_train
        K_inv = self.K_inv
        K_s = self.kernel(X_train, X_test)
        K_ss = self.kernel.diag(X_test)
        mean = K_s.T @ (K_inv @ y_train)
        var = K_ss - np.sum(K_s * (K_inv @ K_s), axis=0)
        return mean, np.maximum(var, 0.0)

    def integral_posterior(self, measure) -> tuple[float, float]:
        self._check_fitted()
        X_train = self.X_train
        y_train = self.y_train
        K_inv = self.K_inv
        mu_f = kernel_mean_vector(X_train, self.kernel, measure)
        sigma_f2 = kernel_integral_variance(self.kernel, measure)
        mean_F = mu_f @ (K_inv @ y_train)
        var_F = sigma_f2 - mu_f @ (K_inv @ mu_f)
        return float(mean_F), float(np.maximum(var_F, 0.0))
