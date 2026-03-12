import numpy as np

from source.functions.interval import Interval
from source.functions.function import Function
from source.numeric_integration.weighted_nodes_numeric_integral import WeightedNodesNumericIntegral


class GaussChebyshevIntegral(WeightedNodesNumericIntegral):
    def __init__(self, func: Function, n: int):
        if n <= 0:
            raise ValueError("n (number of quadrature points) must be positive")
        nodes, weights = np.polynomial.chebyshev.chebgauss(n)
        super().__init__(func, nodes.tolist(), weights.tolist(), Interval(-1.0, 1.0))
