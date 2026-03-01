from abc import abstractmethod
from abc import ABC

from source.data_structures.interval import Interval
from source.data_structures.function import Function


class NumericIntegral(ABC):
    def __init__(self, input_function: Function, input_interval: Interval, input_x_coords: list[float], input_sub_intervals: int) -> None:
        self._interval = input_interval
        self._function = input_function
        self._sub_intervals = input_sub_intervals
        self._nodes = input_x_coords
        self._validate_sub_intervals()
        self._validate_nodes()

    def _validate_sub_intervals(self) -> None:
        if self._sub_intervals <= 0:
            raise RuntimeError("Count of input subintervals must be greater than 0")

    def _validate_nodes(self) -> None:
        nodes = self._nodes
        sub_intervals = self._sub_intervals
        if len(nodes) != sub_intervals + 1:
            raise RuntimeError("input_x_coords must have exactly sub_intervals + 1 nodes")

        left = self._interval.get_left_component()
        right = self._interval.get_right_component()
        if nodes[0] != left or nodes[-1] != right:
            raise RuntimeError("Nodes must start/end at the interval boundaries")

        expected_step = (right - left) / sub_intervals
        for i in range(1, len(nodes)):
            if nodes[i] - nodes[i - 1] != expected_step:
                raise RuntimeError("Nodes must be uniformly spaced across the interval")

    @abstractmethod
    def integrate(self):
        raise RuntimeError("Not implemented yet")