import numpy as np
from typing import Optional, Callable

from source.numeric_integration.monte_carlo.monte_carlo_numeric_integral import MonteCarloNumericIntegral
from source.measures.measure import Measure


class WeightedMonteCarloIntegral(MonteCarloNumericIntegral):
    def __init__(self, input_function, measure: Measure, n_samples: int, rng: Optional[np.random.Generator] = None,
        proposal_sampler: Optional[Callable[[int, np.random.Generator], np.ndarray]] = None, weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        super().__init__(input_function, measure, n_samples, rng)
        self._proposal_sampler = proposal_sampler
        self._weight_fn = weight_fn

    def integrate(self) -> float:
        if self._proposal_sampler is None or self._weight_fn is None:
            raise NotImplementedError("Weighted MC: proposal_sampler/weight_fn are required")

        samples = self._proposal_sampler(self._n_samples, self._rng)
        weights = np.asarray(self._weight_fn(samples), dtype=float)
        values = self._function(samples)
        weighted = weights * values
        mean = float(np.mean(weighted))
        std = float(np.std(weighted, ddof=1))
        self._stderr = std / np.sqrt(self._n_samples)
        return mean
