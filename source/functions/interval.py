from typing import Union


class Interval:
    def __init__(self, left_component: Union[float, int], right_component: Union[float, int]) -> None:
        if left_component > right_component:
            raise RuntimeError("Right component must be greater than left")
        if not isinstance(left_component, (int, float)) or not isinstance(right_component, (int, float)):
            raise RuntimeError("Left and right input components must be numeric")
        self._set_left_component(float(left_component))
        self._set_right_component(float(right_component))

    def _set_left_component(self, left) -> None:
        self._left = left

    def _set_right_component(self, right) -> None:
        self._right = right

    def get_left_component(self) -> Union[float, int]:
        return self._left

    def get_right_component(self) -> Union[float, int]:
        return self._right

    def get_len_of_interval(self) -> float:
        return self._right - self._left

    @property
    def width(self) -> float:
        return self._right - self._left

    def contains(self, input_value: Union[int, float]) -> bool:
        return self._left <= input_value and input_value <= self._right

    def __contains__(self, input_value: Union[int, float]) -> bool:
        return self.contains(input_value)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return self._left == other._left and self._right == other._right

    def __lt__(self, other) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return (self._left, self._right) < (other._left, other._right)

    def __le__(self, other) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return (self._left, self._right) <= (other._left, other._right)

    def __gt__(self, other) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return (self._left, self._right) > (other._left, other._right)

    def __ge__(self, other) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return (self._left, self._right) >= (other._left, other._right)

    def __hash__(self) -> int:
        return hash((self._left, self._right))

    def __repr__(self) -> str:
        return f"Interval({self._left}, {self._right})"

    def __str__(self) -> str:
        return f"[{self._left}, {self._right}]"
