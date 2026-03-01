import math
import numpy as np

from source.data_structures.interval import Interval
from source.measures.measure import Measure
from source.numeric_integration.numeric_integral import NumericIntegral


class MonteCarloNumericIntegral(NumericIntegral):
    def __init__(self, input_function, measure: Measure, n_samples: int, rng: np.random.Generator | None = None):
        super().__init__(input_function, Interval(0.0, 1.0), [0.0, 1.0], 1, validate=False)
        self._measure = measure
        self._n_samples = n_samples
        self._rng = rng or np.random.default_rng()
        self._stderr: float | None = None

    def integrate(self) -> float:
        samples = self._measure.sample(self._n_samples, self._rng)
        values = self._function(samples)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        self._stderr = std / math.sqrt(self._n_samples)
        return mean

    @property
    def stderr(self) -> float | None:
        return self._stderr
