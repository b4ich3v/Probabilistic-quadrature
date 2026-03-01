from abc import abstractmethod
from abc import ABC

from source.data_structures.interval import Interval
from source.data_structures.function import Function


class NumericIntegral(ABC):
    def __init__(self, input_function: Function, input_interval: Interval, input_x_coords: list[float], input_sub_intervals: int) -> None:
        if input_sub_intervals < 0 or len(input_x_coords) == 0:
            raise RuntimeError("Count of input subintervals must be greater than 0 and input_x_coords must have atleast one element")
        self._interval = input_interval
        self._function = input_function
        self._sub_intervals = input_sub_intervals
        self._nodes = input_x_coords

    @abstractmethod
    def integrate(self):
        raise RuntimeError("Not implemented yet")