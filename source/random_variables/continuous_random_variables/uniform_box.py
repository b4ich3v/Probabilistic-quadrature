import numpy as np
from typing import Optional

from source.random_variables.random_variable import RandomVariable


class ContinuousUniformBox(RandomVariable):
    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        lower = np.atleast_1d(lower).astype(float)
        upper = np.atleast_1d(upper).astype(float)
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if np.any(upper <= lower):
            raise ValueError("each component must satisfy upper_i > lower_i")
        self._lower = lower
        self._upper = upper

    @property
    def dim(self) -> int:
        return self._lower.shape[0]

    @property
    def lower(self) -> np.ndarray:
        return self._lower

    @property
    def upper(self) -> np.ndarray:
        return self._upper

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        u = rng.random((n, self.dim))
        return self._lower + u * (self._upper - self._lower)

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError("Input dimensionality mismatch")
        inside = (x >= self._lower) & (x <= self._upper)
        inside = np.all(inside, axis=1)
        vol = np.prod(self._upper - self._lower)
        log_p = np.full(x.shape[0], -np.inf, dtype=float)
        log_p[inside] = -np.log(vol)
        return log_p

    def mean(self) -> np.ndarray:
        return 0.5 * (self._lower + self._upper)

    def var(self) -> np.ndarray:
        return np.square(self._upper - self._lower) / 12.0

    def __repr__(self) -> str:
        return f"ContinuousUniformBox(lower={self._lower!r}, upper={self._upper!r})"
