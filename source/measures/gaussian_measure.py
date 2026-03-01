import numpy as np

from source.measures.measure import Measure


class GaussianMeasure(Measure):
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = np.atleast_1d(mean).astype(float)
        self.cov = np.atleast_2d(cov).astype(float)
        if self.cov.shape[0] != self.cov.shape[1]:
            raise ValueError("covariance must be square")
        if self.cov.shape[0] != self.mean.shape[0]:
            raise ValueError("mean and covariance dimension mismatch")

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.multivariate_normal(self.mean, self.cov, size=n)
