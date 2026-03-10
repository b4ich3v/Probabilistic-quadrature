import numpy as np

from source.numeric_integration.monte_carlo.monte_carlo_numeric_integral import MonteCarloNumericIntegral
from source.random_variables.continuous_random_variables.uniform import Uniform


class RecursiveMonteCarloIntegral(MonteCarloNumericIntegral):
    def __init__(self, input_function, measure, n_samples: int, depth: int = 1, rv=None, rng=None):
        if not isinstance(depth, (int, float)) and rv is None and rng is not None:
            rv, depth = rng, 1
        super().__init__(input_function, measure, n_samples, rv=rv or Uniform(0.0, 1.0))
        self._depth = int(depth)
        if self._depth < 0:
            raise ValueError("depth must be non-negative")
        if self._measure.lower.shape[0] != 1 or self._measure.upper.shape[0] != 1:
            raise ValueError("Recursive Monte Carlo currently supports only 1D measures")

    def integrate(self) -> float:
        min_samples = 1 << self._depth
        if self._n_samples < min_samples:
            raise ValueError(f"n_samples must be >= 2^depth ({min_samples}) for recursive MC")
        return self._integrate_recursive(self._measure.lower[0], self._measure.upper[0], self._depth, self._n_samples)

    def _integrate_recursive(self, a: float, b: float, depth: int, n: int) -> float:
        if n <= 0:
            raise ValueError("Each recursion branch needs at least one sample")
        if depth == 0:
            sampler = Uniform(a, b)
            xs = sampler.sample(n)
            vals = self._function(xs)
            return float(np.mean(vals))

        mid = 0.5 * (a + b)
        left_n = n // 2
        right_n = n - left_n

        left_est = self._integrate_recursive(a, mid, depth - 1, left_n)
        right_est = self._integrate_recursive(mid, b, depth - 1, right_n)

        left_weight = (mid - a) / (b - a)
        right_weight = (b - mid) / (b - a)
        return left_weight * left_est + right_weight * right_est
