import numpy as np
from typing import Optional

from source.measures.measure import Measure
from source.random_variables.continuous_random_variables.uniform_box import ContinuousUniformBox
from source.random_variables.random_variable import RandomVariable


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

    def sample(self, n: int, rv: Optional[RandomVariable] = None) -> np.ndarray:
        return (rv or self._rv).sample(n)
