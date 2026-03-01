import numpy as np

from source.data_structures.interval import Interval
from source.data_structures.function import Function
from source.numeric_integration.weighted_nodes_numeric_integral import WeightedNodesNumericIntegral


class GaussHermiteIntegral(WeightedNodesNumericIntegral):
    def __init__(self, input_function: Function, n: int):
        nodes, weights = np.polynomial.hermite.hermgauss(n)
        super().__init__(input_function, nodes.tolist(), weights.tolist(), Interval(float("-inf"), float("inf")))

    def integrate(self) -> float:
        total = 0.0
        for x, w in zip(self._nodes, self._weights):
            total += w * self._function(x)
        return float(total)
