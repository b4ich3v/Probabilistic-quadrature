from typing import Callable, Sequence, Optional, Union, List

from source.functions.interval import Interval


class Domain:
    def __init__(self, input_intervals: Union[Interval, Sequence[Interval]], input_predicate: Optional[Callable[[float], bool]] = None):
        if isinstance(input_intervals, Interval):
            intervals: list[Interval] = [input_intervals]
        else:
            intervals = list(input_intervals)
        if len(intervals) == 0:
            raise RuntimeError("Domain must have at least one interval")
        self._is_continuous = len(intervals) == 1
        self._predicate = input_predicate
        self._intervals = intervals

    def get_intervals(self) -> Union[List[Interval], Interval]:
        return self._intervals[0] if self._is_continuous else self._intervals
    
    @property
    def intervals(self) -> List[Interval]:
        return self._intervals
    
    def contains(self, input_value: Union[float, int]) -> bool:
        predicate = self._predicate
        if predicate is not None and not predicate(input_value):
            return False

        if self._is_continuous:
            interval = self._intervals[0]
            return interval.contains(input_value)

        return any(interval.contains(input_value) for interval in self._intervals)
