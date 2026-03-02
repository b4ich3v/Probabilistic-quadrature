import numpy as np

from source.measures.measure import Measure
from source.random_variables.continuous_random_variables.normal import Normal


class GaussianMeasure(Measure):
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        mean, cov, std, full_cov, is_diag = self._prepare_params(mean, cov)
        self.mean = mean
        self.cov = cov
        self._full_cov = full_cov if not is_diag else None
        self._rv_diag = Normal(mean, std)

    @staticmethod
    def _prepare_params(mean: np.ndarray, cov: np.ndarray):
        mean = np.atleast_1d(mean).astype(float)
        cov = np.atleast_2d(cov).astype(float)

        if cov.shape[0] != cov.shape[1]:
            raise ValueError("covariance must be square")
        if cov.shape[0] != mean.shape[0]:
            raise ValueError("mean and covariance dimension mismatch")

        is_diag = np.allclose(cov, np.diag(np.diag(cov)))
        std = np.sqrt(np.diag(cov))
        full_cov = cov if not is_diag else None
        return mean, cov, std, full_cov, is_diag

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if self._full_cov is not None:
            return rng.multivariate_normal(self.mean, self.cov, size=n)
        return self._rv_diag.sample(n, rng)
