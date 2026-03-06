import numpy as np
from numpy.typing import ArrayLike
from typing import Optional

from source.kernels.kernel import Kernel
from source.kernels.rbf_kernel import RBFKernel
from source.measures.gaussian_measure import GaussianMeasure


def ensure_2d(X: ArrayLike) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, d)")
    return X


def _gaussian_kernel_mean_rbf(X: np.ndarray, kernel: RBFKernel, measure: GaussianMeasure) -> np.ndarray:
    d = measure.dim
    ell2 = kernel.lengthscale**2
    cov = measure.cov
    mean = measure.mean
    Sigma = cov + ell2 * np.eye(d)
    Sigma_inv = np.linalg.inv(Sigma)
    norm_const = kernel.variance * np.sqrt(
        np.linalg.det(ell2 * np.eye(d)) / np.linalg.det(Sigma)
    )
    diff = X - mean
    exponents = -0.5 * np.einsum("ni,ij,nj->n", diff, Sigma_inv, diff)
    return norm_const * np.exp(exponents)


def _gaussian_kernel_variance_rbf(kernel: RBFKernel, measure: GaussianMeasure) -> float:
    d = measure.dim
    ell2 = kernel.lengthscale**2
    Sigma = measure.cov
    Sigma2 = 2.0 * Sigma
    Sigma_eff = Sigma2 + ell2 * np.eye(d)
    norm_const = kernel.variance * np.sqrt(
        np.linalg.det(ell2 * np.eye(d)) / np.linalg.det(Sigma_eff)
    )
    return float(norm_const)


def kernel_mean_vector(X: ArrayLike, kernel: Kernel, measure, mc_samples: int = 2048, rng=None) -> np.ndarray:
    X = ensure_2d(X)
    if isinstance(kernel, RBFKernel) and isinstance(measure, GaussianMeasure):
        return _gaussian_kernel_mean_rbf(X, kernel, measure)

    samples = measure.sample(mc_samples)
    K = kernel(samples, X)
    weights = np.full(mc_samples, 1.0 / mc_samples)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        return K.T @ weights


def kernel_integral_variance(kernel, measure, mc_samples: int = 4096, rng=None) -> float:
    if isinstance(kernel, RBFKernel) and isinstance(measure, GaussianMeasure):
        return _gaussian_kernel_variance_rbf(kernel, measure)

    rng = rng or np.random.default_rng()
    samples = measure.sample(mc_samples)
    K = kernel(samples, samples)
    np.fill_diagonal(K, 0.0)
    n = samples.shape[0]
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        return float(K.sum() / (n * (n - 1)))


def gp_posterior_predictive(X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike, kernel, noise: float = 0.0, K_inv: Optional[np.ndarray] = None):
    X_train = ensure_2d(X_train)
    X_test = ensure_2d(X_test)
    y_train = np.asarray(y_train, dtype=float)
    K = kernel(X_train, X_train)
    if noise > 0.0:
        K = K + (noise**2) * np.eye(len(X_train))
    if K_inv is None:
        K_inv = np.linalg.solve(K, np.eye(K.shape[0]))

    K_s = kernel(X_train, X_test)
    K_ss = kernel.diag(X_test)
    mean = K_s.T @ (K_inv @ y_train)
    var = K_ss - np.sum(K_s * (K_inv @ K_s), axis=0)
    return mean, np.maximum(var, 0.0)
