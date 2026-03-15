import numpy as np

from source.functions.interval import Interval
from source.functions.function import Function
from source.numeric_integration.weighted_nodes_numeric_integral import WeightedNodesNumericIntegral


# Gauss-Hermite quadrature for integrals weighted by exp(-x^2) on (-inf, inf)
class GaussHermiteIntegral(WeightedNodesNumericIntegral):
    def __init__(self, func: Function, n: int):
        if n <= 0:
            raise ValueError("n (number of quadrature points) must be positive")
        nodes, weights = np.polynomial.hermite.hermgauss(n)
        super().__init__(func, nodes.tolist(), weights.tolist(), Interval(float("-inf"), float("inf")))