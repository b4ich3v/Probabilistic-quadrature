import numpy as np

from source.measures.measure import Measure


class UniformBoxMeasure(Measure):
    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        lower = np.atleast_1d(lower).astype(float)
        upper = np.atleast_1d(upper).astype(float)
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if np.any(upper <= lower):
            raise ValueError("each component must satisfy upper_i > lower_i")
        self.lower = lower
        self.upper = upper

    @property
    def dim(self) -> int:
        return self.lower.shape[0]

    def volume(self) -> float:
        return float(np.prod(self.upper - self.lower))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        u = rng.random((n, self.dim))
        return self.lower + u * (self.upper - self.lower)
