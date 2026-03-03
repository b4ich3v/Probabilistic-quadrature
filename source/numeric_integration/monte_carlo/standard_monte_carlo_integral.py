import math
import numpy as np

from source.numeric_integration.monte_carlo.monte_carlo_numeric_integral import MonteCarloNumericIntegral


class StandardMonteCarloIntegral(MonteCarloNumericIntegral):
    def integrate(self) -> float:
        samples = self._measure.sample(self._n_samples, self._rng)
        values = self._function(samples)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        self._stderr = std / math.sqrt(self._n_samples)
        return mean