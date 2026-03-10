from typing import Optional

from source.functions.interval import Interval
from source.measures.measure import Measure
from source.numeric_integration.numeric_integral import NumericIntegral
from source.random_variables.random_variable import RandomVariable


class MonteCarloNumericIntegral(NumericIntegral):
    def __init__(self, input_function, measure: Measure, n_samples: int, rv: Optional[RandomVariable] = None, rng=None) -> None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        super().__init__(input_function, Interval(0.0, 1.0), [0.0, 1.0], 1, validate=False)
        self._measure = measure
        self._n_samples = n_samples
        self._rv = rv
        self._stderr: Optional[float] = None

    @property
    def stderr(self) -> Optional[float]:
        return self._stderr

    def integrate(self) -> float:
        raise NotImplementedError("Subclass responsibility")