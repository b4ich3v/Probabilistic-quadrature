from typing import Callable, Sequence

from source.data_structures.interval import Interval


class Domain:
    def __init__(self, input_intervals: Interval | Sequence[Interval], input_predicate: Callable[[float], bool] | None = None):
        if isinstance(input_intervals, Interval):
            intervals: list[Interval] = [input_intervals]
        else:
            intervals = list(input_intervals)
        if len(intervals) == 0:
            raise RuntimeError("Domain must have at least one interval")
        self.__is_continuous = len(intervals) == 1
        self.__predicate = input_predicate
        self.__intervals = intervals

    def get_intervals(self) -> list[Interval] | Interval:
        return self.__intervals[0] if self.__is_continuous else self.__intervals
    
    @property
    def intervals(self) -> list[Interval]:
        return self.__intervals
    
    def contains(self, input_value: float | int) -> bool:
        predicate = self.__predicate
        if predicate is not None and not predicate(input_value):
            return False

        if self.__is_continuous:
            interval = self.__intervals[0]
            return interval.contains(input_value)

        return any(interval.contains(input_value) for interval in self.__intervals)