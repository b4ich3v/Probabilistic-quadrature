import numpy as np
from numpy.typing import ArrayLike
from typing import Optional
from scipy.linalg import cho_factor, cho_solve

from source.kernels.kernel import Kernel
from source.kernels.rbf_kernel import RBFKernel
from source.measures.gaussian_measure import GaussianMeasure


# Normalize input to shape (n_samples, d)
def ensure_2d(X: ArrayLike) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, d)")
    return X


# Closed-form kernel mean E_mu[k(x, .)] for RBF kernel + Gaussian measure
# Uses slogdet for numerical stability in high dimensions
def _gaussian_kernel_mean_rbf(X: np.ndarray, kernel: RBFKernel, measure: GaussianMeasure) -> np.ndarray:
    d = measure.dim
    ell2 = kernel.lengthscale ** 2
    cov = measure.cov
    mean = measure.mean
    Sigma = cov + ell2 * np.eye(d)  # Sigma_measure + ell^2 * I

    # Log-space normalization: log(sigma^2) + 0.5*(logdet(ell^2*I) - logdet(Sigma))
    sign, logdet_Sigma = np.linalg.slogdet(Sigma)
    _, logdet_ell = np.linalg.slogdet(ell2 * np.eye(d))
    log_norm = np.log(kernel.variance) + 0.5 * (logdet_ell - logdet_Sigma)

    # Mahalanobis distance via Cholesky solve
    L_sigma = np.linalg.cholesky(Sigma)
    diff = X - mean
    alpha = np.linalg.solve(L_sigma, diff.T)
    exponents = -0.5 * np.sum(alpha ** 2, axis=0)
    return np.exp(log_norm + exponents)


# Closed-form integrated kernel variance E_mu[E_mu[k(., .)]] for RBF + Gaussian
def _gaussian_kernel_variance_rbf(kernel: RBFKernel, measure: GaussianMeasure) -> float:
    d = measure.dim
    ell2 = kernel.lengthscale ** 2
    Sigma = measure.cov
    Sigma_eff = 2.0 * Sigma + ell2 * np.eye(d)  # effective covariance from double integral

    _, logdet_ell = np.linalg.slogdet(ell2 * np.eye(d))
    sign, logdet_eff = np.linalg.slogdet(Sigma_eff)
    log_norm = np.log(kernel.variance) + 0.5 * (logdet_ell - logdet_eff)
    return float(np.exp(log_norm))


# mu_f(x_i) = E_mu[k(x_i, .)]; analytic for RBF+Gaussian, MC fallback otherwise
def kernel_mean_vector(X: ArrayLike, kernel: Kernel, measure, mc_samples: int = 2048,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    X = ensure_2d(X)
    if isinstance(kernel, RBFKernel) and isinstance(measure, GaussianMeasure):
        return _gaussian_kernel_mean_rbf(X, kernel, measure)

    # MC approximation: (1/M) * sum_j k(z_j, x_i) where z_j ~ mu
    samples = measure.sample(mc_samples, rng=rng)
    K = kernel(samples, X)
    weights = np.full(mc_samples, 1.0 / mc_samples)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        return K.T @ weights


# sigma_f^2 = E_mu[E_mu[k(., .)]]; analytic for RBF+Gaussian, MC fallback otherwise
def kernel_integral_variance(kernel: Kernel, measure, mc_samples: int = 4096,
                             rng: np.random.Generator | None = None) -> float:
    if isinstance(kernel, RBFKernel) and isinstance(measure, GaussianMeasure):
        return _gaussian_kernel_variance_rbf(kernel, measure)

    # U-statistic estimator: sum off-diagonal K(z_i, z_j) / (n*(n-1))
    samples = measure.sample(mc_samples, rng=rng)
    K = kernel(samples, samples)
    np.fill_diagonal(K, 0.0)
    n = samples.shape[0]
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        est = float(K.sum() / (n * (n - 1)))
        return max(est, 0.0)


# Standalone GP predictive mean and variance; reuses L_factor if provided
def gp_posterior_predictive(
    X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike,
    kernel: Kernel, noise: float = 0.0, L_factor: Optional[tuple] = None) -> tuple[np.ndarray, np.ndarray]:
    X_train = ensure_2d(X_train)
    X_test = ensure_2d(X_test)
    y_train = np.asarray(y_train, dtype=float)

    if L_factor is None:
        K = kernel(X_train, X_train)
        if noise > 0.0:
            K = K + (noise ** 2) * np.eye(len(X_train))
        try:
            L_factor = cho_factor(K)
        except np.linalg.LinAlgError:
            raise ValueError("Kernel matrix is not positive definite. Try increasing jitter or noise.")

    K_s = kernel(X_train, X_test)
    alpha = cho_solve(L_factor, y_train)
    mean = K_s.T @ alpha
    v = cho_solve(L_factor, K_s)
    var = kernel.diag(X_test) - np.sum(K_s * v, axis=0)
    return mean, np.maximum(var, 0.0)
