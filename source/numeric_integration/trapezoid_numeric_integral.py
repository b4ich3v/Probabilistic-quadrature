from source.numeric_integration.numeric_integral import NumericIntegral
from source.data_structures.interval import Interval
from source.data_structures.function import Function


class TrapezoidNumericIntegral(NumericIntegral):
    def __init__(self, input_function: Function, input_interval: Interval, input_x_coords: list[float], input_sub_intervals: int):
        super().__init__(input_function, input_interval, input_x_coords, input_sub_intervals)

    def integrate(self):
        total = 0.0
        nodes = self._nodes
        f = self._function
        for i in range(self._sub_intervals):
            total += f(nodes[i]) + f(nodes[i + 1])
        return total * (self._interval.width / (2 * self._sub_intervals))
