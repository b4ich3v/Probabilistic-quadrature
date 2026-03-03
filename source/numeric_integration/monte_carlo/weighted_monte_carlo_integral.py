import numpy as np
from typing import Optional

from source.numeric_integration.monte_carlo.monte_carlo_numeric_integral import MonteCarloNumericIntegral
from source.measures.measure import Measure


class WeightedMonteCarloIntegral(MonteCarloNumericIntegral):
    def __init__(self, input_function, measure: Measure, n_samples: int, rng: Optional[np.random.Generator] = None, proposal=None):
        super().__init__(input_function, measure, n_samples, rng)
        self._proposal = proposal

    def integrate(self) -> float:
        raise NotImplementedError("Weighted MC requires proposal distribution with sample/log_prob")