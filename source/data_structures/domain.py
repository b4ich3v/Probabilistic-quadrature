from typing import Callable

from source.data_structures.interval import Interval


class Domain:
    def __init__(self, input_intervals: list[Interval] | Interval | None, input_predicate: Callable[[float], bool] | None = None):
        if len(input_intervals) == 0:
            raise RuntimeError("Domain must have at least one interval")
        if self.__determine_continuously(input_intervals):
            self.__is_continuous = True
        else: 
            self.__is_continuous = False
        self.__predicate = input_predicate
        self.__intervals = input_intervals

    def __determine_continuously(self, input_intervals) -> bool:
        return len(input_intervals) == 1

    def get_intervals(self) -> list[Interval] | Interval:
        if self.__is_continuous:
            return self.__intervals[0]
        else:
            return self.__intervals
    
    def contains(self, input_value: float | int) -> bool:
        predicate = self.__predicate
        if predicate is not None and not predicate(input_value):
            return False

        if self.__is_continuous:
            interval = self.__intervals[0]
            return interval.contains(input_value)

        for interval in self.__intervals:
            if interval.contains(input_value):
                return True
        return False