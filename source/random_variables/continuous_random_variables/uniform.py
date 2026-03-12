import numpy as np

from source.random_variables.random_variable import RandomVariable


class Uniform(RandomVariable):
    def __init__(self, low: float, high: float):
        if high <= low:
            raise ValueError("high must be > low")
        self._low = float(low)
        self._high = float(high)

    @property
    def dim(self) -> int:
        return 1

    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        u = rng.random((n, 1))
        return self._low + u * (self._high - self._low)

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        if x.shape[1] != 1:
            raise ValueError("Input dimensionality mismatch")
        inside = (self._low <= x[:, 0]) & (x[:, 0] <= self._high)
        log_p = np.full(x.shape[0], -np.inf, dtype=float)
        width = self._high - self._low
        log_p[inside] = -np.log(width)
        return log_p

    def mean(self) -> np.ndarray:
        return np.array([(self._low + self._high) * 0.5])

    def var(self) -> np.ndarray:
        width = self._high - self._low
        return np.array([width * width / 12.0])

    def __repr__(self) -> str:
        return f"Uniform(low={self._low}, high={self._high})"
