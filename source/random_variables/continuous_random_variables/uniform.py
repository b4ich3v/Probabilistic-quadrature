import numpy as np
from typing import Optional

from source.random_variables.random_variable import RandomVariable


class Uniform(RandomVariable):
    def __init__(self, low: int, high: int):
        if high < low:
            raise ValueError("high must be >= low")
        self._low = int(low)
        self._high = int(high)
        self._count = self._high - self._low + 1

    @property
    def dim(self) -> int:
        return 1

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.integers(self._low, self._high + 1, size=(n, 1))

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        if x.shape[1] != 1:
            raise ValueError("Input dimensionality mismatch")
        inside = (self._low <= x[:, 0]) & (x[:, 0] <= self._high)
        log_p = np.full(x.shape[0], -np.inf, dtype=float)
        log_p[inside] = -np.log(self._count)
        return log_p

    def mean(self) -> np.ndarray:
        return np.array([(self._low + self._high) / 2.0])

    def var(self) -> np.ndarray:
        return np.array([((self._count ** 2) - 1) / 12.0])

    def __repr__(self) -> str:
        return f"DiscreteUniform(low={self._low}, high={self._high})"
