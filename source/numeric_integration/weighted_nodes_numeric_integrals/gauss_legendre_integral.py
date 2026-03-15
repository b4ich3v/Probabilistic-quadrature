import numpy as np

from source.functions.interval import Interval
from source.functions.function import Function
from source.numeric_integration.weighted_nodes_numeric_integral import WeightedNodesNumericIntegral
from source.numeric_integration.weighted_nodes_numeric_integrals.affine_transformation import AffineTransformation


# Gauss-Legendre quadrature; supports arbitrary [a,b] via affine mapping
class GaussLegendreIntegral(WeightedNodesNumericIntegral):
    def __init__(self, func: Function, n: int, interval: Interval | None = None):
        if n <= 0:
            raise ValueError("n (number of quadrature points) must be positive")
        nodes, weights = np.polynomial.legendre.leggauss(n)
        target_interval = interval or Interval(-1.0, 1.0)
        if interval is not None:
            nodes, weights = AffineTransformation.map_from_unit(nodes, weights, target_interval)
        super().__init__(func, nodes.tolist(), weights.tolist(), target_interval)
