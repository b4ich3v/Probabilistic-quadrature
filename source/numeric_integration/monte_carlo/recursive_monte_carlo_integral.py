import numpy as np
from typing import Callable

from source.numeric_integration.monte_carlo.monte_carlo_numeric_integral import MonteCarloNumericIntegral
from source.measures.uniform_box_measure import UniformBoxMeasure
from source.random_variables.continuous_random_variables.uniform import Uniform


class RecursiveMonteCarloIntegral(MonteCarloNumericIntegral):
    def __init__(self, func: Callable, measure: UniformBoxMeasure, n_samples: int, depth: int = 1, **kwargs):
        if not isinstance(measure, UniformBoxMeasure):
            raise TypeError("Recursive Monte Carlo requires a UniformBoxMeasure")
        super().__init__(func, measure, n_samples, rv=kwargs.get("rv"))
        self._depth = int(depth)
        if self._depth < 0:
            raise ValueError("depth must be non-negative")
        if measure.lower.shape[0] != 1 or measure.upper.shape[0] != 1:
            raise ValueError("Recursive Monte Carlo currently supports only 1D measures")

    def integrate(self) -> float:
        min_samples = 1 << self._depth
        if self._n_samples < min_samples:
            raise ValueError(f"n_samples must be >= 2^depth ({min_samples}) for recursive MC")
        a = float(self._measure.lower[0])
        b = float(self._measure.upper[0])
        return self._integrate_recursive(a, b, self._depth, self._n_samples)

    def _integrate_recursive(self, a: float, b: float, depth: int, n: int) -> float:
        if n <= 0:
            raise ValueError("Each recursion branch needs at least one sample")
        if depth == 0:
            sampler = Uniform(a, b)
            xs = sampler.sample(n)
            vals = self._function(xs)
            return float(np.mean(vals)) * (b - a)

        mid = 0.5 * (a + b)
        left_n = n // 2
        right_n = n - left_n

        left_est = self._integrate_recursive(a, mid, depth - 1, left_n)
        right_est = self._integrate_recursive(mid, b, depth - 1, right_n)

        return left_est + right_est