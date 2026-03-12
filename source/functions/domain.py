from __future__ import annotations

from typing import Callable, Optional, Sequence

from source.functions.interval import Interval


class Domain:
    def __init__(self, intervals: Interval | Sequence[Interval], predicate: Callable[[float], bool] | None = None):
        if isinstance(intervals, Interval):
            interval_list: list[Interval] = [intervals]
        else:
            interval_list = list(intervals)
        if len(interval_list) == 0:
            raise ValueError("Domain must have at least one interval")
        self._is_continuous = len(interval_list) == 1
        self._predicate = predicate
        self._intervals = interval_list

    @property
    def intervals(self) -> list[Interval]:
        return list(self._intervals)

    @property
    def interval(self) -> Interval:
        if not self._is_continuous:
            raise ValueError("Domain has multiple intervals; use .intervals")
        return self._intervals[0]

    def contains(self, value: float | int) -> bool:
        if self._predicate is not None and not self._predicate(value):
            return False
        if self._is_continuous:
            return self._intervals[0].contains(value)
        return any(iv.contains(value) for iv in self._intervals)
