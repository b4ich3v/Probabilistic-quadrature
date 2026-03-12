import numpy as np

from source.random_variables.random_variable import RandomVariable


class Normal(RandomVariable):
    def __init__(self, mean: float | np.ndarray, std: float | np.ndarray):
        mu = np.atleast_1d(mean).astype(float)
        std_arr = np.atleast_1d(std).astype(float)
        if std_arr.shape not in [(1,), mu.shape]:
            raise ValueError("std must be scalar or match mean shape")
        if np.any(std_arr <= 0):
            raise ValueError("std must be positive")
        if std_arr.shape == (1,):
            std_arr = np.full_like(mu, std_arr.item())
        self._mean = mu
        self._std = std_arr

    @property
    def dim(self) -> int:
        return self._mean.shape[0]

    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.normal(loc=self._mean, scale=self._std, size=(n, self.dim))

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(float)
        if x.shape[1] != self.dim:
            raise ValueError("Input dimensionality mismatch")
        diff = (x - self._mean) / self._std
        log_det = np.log(self._std).sum()
        quad = 0.5 * np.sum(diff * diff, axis=1)
        norm_const = 0.5 * self.dim * np.log(2 * np.pi) + log_det
        return -(norm_const + quad)

    def mean(self) -> np.ndarray:
        return self._mean.copy()

    def var(self) -> np.ndarray:
        return np.square(self._std)

    def __repr__(self) -> str:
        return f"Normal(mean={self._mean!r}, std={self._std!r})"
