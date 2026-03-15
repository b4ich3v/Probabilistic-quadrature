import numpy as np

from source.numeric_integration.numeric_integral import NumericIntegral
from source.functions.interval import Interval
from source.functions.function import Function


# Base for quadrature rules with precomputed nodes and weights
class WeightedNodesNumericIntegral(NumericIntegral):
    def __init__(self, func: Function, nodes: list[float], weights: list[float], interval: Interval) -> None:
        if len(nodes) == 0:
            raise ValueError("nodes must not be empty")
        if len(nodes) != len(weights):
            raise ValueError("nodes and weights must have the same length")
        self._function = func
        self._interval = interval
        self._nodes = nodes
        self._weights = weights

    # Integral ≈ sum_i w_i * f(x_i)
    def integrate(self) -> float:
        return float(np.dot(self._weights, [self._function(x) for x in self._nodes]))