import numpy as np
from typing import Callable

from source.numeric_integration.monte_carlo.monte_carlo_numeric_integral import MonteCarloNumericIntegral
from source.measures.measure import Measure
from source.random_variables.random_variable import RandomVariable
from source.random_variables.continuous_random_variables.uniform import Uniform


# Importance-sampling MC: E[f] ≈ (1/n) * sum w(x_i) * f(x_i)
class WeightedMonteCarloIntegral(MonteCarloNumericIntegral):
    def __init__(self, func: Callable, measure: Measure, n_samples: int,
        proposal_sampler: Callable[[int, RandomVariable], np.ndarray], weight_fn: Callable[[np.ndarray], np.ndarray], rv: RandomVariable | None = None):
        super().__init__(func, measure, n_samples, rv)
        self._proposal_sampler = proposal_sampler
        self._weight_fn = weight_fn

    def _default_rv(self) -> RandomVariable:
        return self._rv if self._rv is not None else Uniform(0.0, 1.0)

    def integrate(self) -> float:
        rv = self._default_rv()
        samples = self._proposal_sampler(self._n_samples, rv)
        weights = np.asarray(self._weight_fn(samples), dtype=float)
        values = self._function(samples)
        weighted = weights * values
        mean = float(np.mean(weighted))
        std = float(np.std(weighted, ddof=1))
        self._stderr = std / np.sqrt(self._n_samples)
        return mean