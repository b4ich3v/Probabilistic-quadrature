import math
import numpy as np

from source.numeric_integration.monte_carlo.monte_carlo_numeric_integral import MonteCarloNumericIntegral


# Simple sample-average estimator: E[f] ≈ (1/n) * sum f(x_i), O(1/sqrt(n))
class StandardMonteCarloIntegral(MonteCarloNumericIntegral):
    def integrate(self) -> float:
        samples = self._measure.sample(self._n_samples)
        values = self._function(samples)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))  # unbiased std for stderr
        self._stderr = std / math.sqrt(self._n_samples)
        return mean