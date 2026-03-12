from typing import Optional, Callable
import numpy as np

from source.measures.measure import Measure
from source.numeric_integration.numeric_integral import NumericIntegral
from source.random_variables.random_variable import RandomVariable


class MonteCarloNumericIntegral(NumericIntegral):
    def __init__(self, func: Callable, measure: Measure, n_samples: int,
                 rv: Optional[RandomVariable] = None) -> None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        self._function = func
        self._measure = measure
        self._n_samples = n_samples
        self._rv = rv
        self._stderr: Optional[float] = None

    @property
    def stderr(self) -> Optional[float]:
        return self._stderr

    def integrate(self) -> float:
        raise NotImplementedError("Subclass responsibility")