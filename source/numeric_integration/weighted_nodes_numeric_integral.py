
from source.numeric_integration.numeric_integral import NumericIntegral
from source.functions.interval import Interval
from source.functions.function import Function


class WeightedNodesNumericIntegral(NumericIntegral):
    def __init__(self, input_function: Function, nodes: list[float], weights: list[float], input_interval: Interval) -> None:
        if len(nodes) != len(weights):
            raise RuntimeError("nodes and weights must have the same length")
        super().__init__(input_function, input_interval, nodes, len(nodes) - 1, validate=False)
        self._weights = weights
