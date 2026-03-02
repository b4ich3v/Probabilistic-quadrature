import numpy as np
from typing import Callable

from source.data_structures.domain import Domain
from source.data_structures.interval import Interval
from source.data_structures.function import Function
from source.measures.uniform_box_measure import UniformBoxMeasure


class MeasuredFunction(Function):
    def __init__(self, input_function: Callable[[np.ndarray], np.ndarray], measure, true_integral: float | None = None, input_name: str = "f") -> None:
        super().__init__(input_function, Domain(Interval(float("-inf"), float("inf"))), None, input_name)
        self._measure = measure
        self._true_integral = true_integral

    @property
    def measure(self):
        return self._measure

    @property
    def true_integral(self) -> float | None:
        return self._true_integral

    def __call__(self, X):
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