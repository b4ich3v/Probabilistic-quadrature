import numpy as np
from typing import Callable, Optional

from source.measures.measure import Measure
from source.measures.uniform_box_measure import UniformBoxMeasure


class MeasuredFunction:
    """A callable tied to a probability measure, used for probabilistic integration."""

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], measure: Measure,
                 true_integral: Optional[float] = None, name: str = "f") -> None:
        self._function = func
        self._measure = measure
        self._true_integral = true_integral
        self._name = name

    @property
    def measure(self) -> Measure:
        return self._measure

    @property
    def true_integral(self) -> Optional[float]:
        return self._true_integral

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        X2 = np.atleast_2d(X)

        if isinstance(self._measure, UniformBoxMeasure):
            lower = self._measure.lower
            upper = self._measure.upper
            if lower.shape[0] != X2.shape[1]:
                raise ValueError("Input dimensionality does not match measure")
            if np.any(X2 < lower) or np.any(X2 > upper):
                raise ValueError("Input outside measure support")
        return self._function(X2)
