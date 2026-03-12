from source.numeric_integration.uniform_grid_numeric_integral import UniformGridNumericIntegral
from source.functions.interval import Interval
from source.functions.function import Function


class RectangleNumericIntegral(UniformGridNumericIntegral):
    def __init__(self, func: Function, interval: Interval, nodes: list[float], sub_intervals: int):
        super().__init__(func, interval, nodes, sub_intervals)

    def integrate(self) -> float:
        total = 0.0
        nodes = self._nodes
        f = self._function
        for i in range(self._sub_intervals):
            mid = 0.5 * (nodes[i] + nodes[i + 1])
            total += f(mid)
        return total * (self._interval.width / self._sub_intervals)