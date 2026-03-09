import numpy as np

from source.functions.interval import Interval
from source.functions.function import Function
from source.numeric_integration.weighted_nodes_numeric_integral import WeightedNodesNumericIntegral
from source.numeric_integration.weighted_nodes_numeric_integrals.affine_transformation import AffineTransformation


class GaussLegendreIntegral(WeightedNodesNumericIntegral):
    def __init__(self, input_function: Function, n: int, interval: Interval | None = None):
        nodes, weights = np.polynomial.legendre.leggauss(n)
        target_interval = interval or Interval(-1.0, 1.0)
        if interval is not None:
            nodes, weights = AffineTransformation.map_from_unit(nodes, weights, target_interval)
        super().__init__(input_function, nodes.tolist(), weights.tolist(), target_interval)

    def integrate(self) -> float:
        total = 0.0
        for x, w in zip(self._nodes, self._weights):
            total += w * self._function(x)
        return float(total)
