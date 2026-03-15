import numpy as np

from source.measures.measure import Measure
from source.random_variables.continuous_random_variables.uniform_box import ContinuousUniformBox


# Uniform measure over a d-dimensional hyper-rectangle [lower, upper]
class UniformBoxMeasure(Measure):
    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        rv = ContinuousUniformBox(lower, upper)
        self._rv = rv
        self.lower = rv.lower
        self.upper = rv.upper

    @property
    def dim(self) -> int:
        return self._rv.dim

    def volume(self) -> float:
        return float(np.prod(self.upper - self.lower))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        return self._rv.sample(n, rng=rng)
