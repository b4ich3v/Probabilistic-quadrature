import numpy as np
from typing import Optional

from source.numeric_integration.monte_carlo.monte_carlo_numeric_integral import MonteCarloNumericIntegral
from source.random_variables.continuous_random_variables.uniform import Uniform


class RecursiveMonteCarloIntegral(MonteCarloNumericIntegral):
    def __init__(self, input_function, measure, n_samples: int, depth: int = 1, rng: Optional[np.random.Generator] = None):
        if not isinstance(depth, (int, float)) and rng is None:
            rng, depth = depth, 1
        super().__init__(input_function, measure, n_samples, rng)
        self._depth = int(depth)

    def integrate(self) -> float:
        return self._integrate_recursive(self._measure.lower[0], self._measure.upper[0], self._depth, self._n_samples)

    def _integrate_recursive(self, a: float, b: float, depth: int, n: int) -> float:
        if depth == 0:
            sampler = Uniform(a, b)
            xs = sampler.sample(n, self._rng)
            vals = self._function(xs)
            return float(np.mean(vals) * (b - a))
        mid = 0.5 * (a + b)
        return self._integrate_recursive(a, mid, depth - 1, n // 2) + self._integrate_recursive(mid, b, depth - 1, n - n // 2)
