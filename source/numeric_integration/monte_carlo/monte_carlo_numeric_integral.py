import numpy as np
from typing import Optional

from source.functions.interval import Interval
from source.measures.measure import Measure
from source.numeric_integration.numeric_integral import NumericIntegral


class MonteCarloNumericIntegral(NumericIntegral):
    def __init__(self, input_function, measure: Measure, n_samples: int, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(input_function, Interval(0.0, 1.0), [0.0, 1.0], 1, validate=False)
        self._measure = measure
        self._n_samples = n_samples
        self._rng = rng or np.random.default_rng()
        self._stderr: Optional[float] = None

    @property
    def stderr(self) -> Optional[float]:
        return self._stderr

    def integrate(self) -> float:
        raise NotImplementedError("Subclass responsibility")