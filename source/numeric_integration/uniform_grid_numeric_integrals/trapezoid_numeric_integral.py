from source.numeric_integration.uniform_grid_numeric_integral import UniformGridNumericIntegral
from source.functions.interval import Interval
from source.functions.function import Function


class TrapezoidNumericIntegral(UniformGridNumericIntegral):
    def __init__(self, func: Function, interval: Interval, nodes: list[float], sub_intervals: int):
        super().__init__(func, interval, nodes, sub_intervals)

    def integrate(self) -> float:
        nodes = self._nodes
        f = self._function
        n = self._sub_intervals
        f_values = [f(x) for x in nodes]
        total = f_values[0] + f_values[-1]
        for i in range(1, n):
            total += 2.0 * f_values[i]
        return total * (self._interval.width / (2 * n))