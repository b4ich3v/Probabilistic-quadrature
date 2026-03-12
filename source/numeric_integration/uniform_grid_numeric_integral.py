import numpy as np

from source.numeric_integration.numeric_integral import NumericIntegral
from source.functions.interval import Interval
from source.functions.function import Function


class UniformGridNumericIntegral(NumericIntegral):
    def __init__(self, func: Function, interval: Interval, nodes: list[float], sub_intervals: int) -> None:
        if sub_intervals <= 0:
            raise ValueError("sub_intervals must be greater than 0")
        if len(nodes) != sub_intervals + 1:
            raise ValueError("nodes must have exactly sub_intervals + 1 elements")

        left = interval.left
        right = interval.right
        if not np.isclose(nodes[0], left) or not np.isclose(nodes[-1], right):
            raise ValueError("Nodes must start/end at the interval boundaries")

        expected_step = (right - left) / sub_intervals
        for i in range(1, len(nodes)):
            if not np.isclose(nodes[i] - nodes[i - 1], expected_step):
                raise ValueError("Nodes must be uniformly spaced across the interval")

        self._function = func
        self._interval = interval
        self._nodes = nodes
        self._sub_intervals = sub_intervals