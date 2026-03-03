import numpy as np
from typing import Optional, Callable

from source.numeric_integration.monte_carlo.monte_carlo_numeric_integral import MonteCarloNumericIntegral
from source.measures.measure import Measure
from source.random_variables.random_variable import RandomVariable
from source.random_variables.continuous_random_variables.uniform import Uniform


class WeightedMonteCarloIntegral(MonteCarloNumericIntegral):
    def __init__(self, input_function, measure: Measure, n_samples: int, rv: Optional[RandomVariable] = None,
        proposal_sampler: Optional[Callable[[int, RandomVariable], np.ndarray]] = None, weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        super().__init__(input_function, measure, n_samples, rv)
        self._proposal_sampler = proposal_sampler
        self._weight_fn = weight_fn

    def _default_rv(self) -> RandomVariable:
        return self._rv if self._rv is not None else Uniform(0.0, 1.0)

    def integrate(self) -> float:
        if self._proposal_sampler is None or self._weight_fn is None:
            raise NotImplementedError("Weighted MC: proposal_sampler/weight_fn are required")

        rv = self._default_rv()
        samples = self._proposal_sampler(self._n_samples, rv)
        weights = np.asarray(self._weight_fn(samples), dtype=float)
        values = self._function(samples)
        weighted = weights * values
        mean = float(np.mean(weighted))
        std = float(np.std(weighted, ddof=1))
        self._stderr = std / np.sqrt(self._n_samples)
        return mean
