from source.numeric_integration.numeric_integral import NumericIntegral
from source.functions.interval import Interval
from source.functions.function import Function


class UniformGridNumericIntegral(NumericIntegral):
    def __init__(self, input_function: Function, input_interval: Interval, input_x_coords: list[float], input_sub_intervals: int) -> None:
        super().__init__(input_function, input_interval, input_x_coords, input_sub_intervals, validate=True)