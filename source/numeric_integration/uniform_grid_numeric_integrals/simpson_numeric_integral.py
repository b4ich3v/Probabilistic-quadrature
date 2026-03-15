from source.numeric_integration.uniform_grid_numeric_integral import UniformGridNumericIntegral
from source.functions.interval import Interval
from source.functions.function import Function


# Composite Simpson's rule using midpoints, O(h^4) convergence
class SimpsonNumericIntegral(UniformGridNumericIntegral):
    def __init__(self, func: Function, interval: Interval, nodes: list[float], sub_intervals: int):
        super().__init__(func, interval, nodes, sub_intervals)

    # h/6 * [f(x_0) + 4*sum(f(mid_i)) + 2*sum(f(x_i)) + f(x_n)]
    def integrate(self) -> float:
        nodes = self._nodes
        f = self._function
        n = self._sub_intervals
        f_values = [f(x) for x in nodes]
        total = f_values[0] + f_values[-1]
        for i in range(1, n):
            total += 2.0 * f_values[i]
        for i in range(n):
            mid = 0.5 * (nodes[i] + nodes[i + 1])
            total += 4.0 * f(mid)
        return total * (self._interval.width / (6 * n))