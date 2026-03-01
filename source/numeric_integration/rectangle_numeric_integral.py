from source.numeric_integration.numeric_integral import NumericIntegral
from source.data_structures.interval import Interval


class RectangleNumericIntegral(NumericIntegral):
    def __init__(self, input_interval: Interval, input_x_coords: list[float], input_sub_intervals: int):
        super().__init__(input_interval, input_x_coords, input_sub_intervals)

    def integrate(self):
        result = 0
        for i in range(1, self._sub_intervals + 1):
            result += self._function((self._nodes[i - 1] + self._nodes[i]) / 2)
        result *= (self._interval.width) / self._sub_intervals
        return result