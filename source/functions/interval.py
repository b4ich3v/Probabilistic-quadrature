# Closed interval [left, right] on the real line
class Interval:
    def __init__(self, left: float | int, right: float | int) -> None:
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise TypeError("Left and right components must be numeric")
        if left > right:
            raise ValueError("Right component must be greater than or equal to left")
        self._left = float(left)
        self._right = float(right)

    @property
    def left(self) -> float:
        return self._left

    @property
    def right(self) -> float:
        return self._right

    @property
    def width(self) -> float:
        return self._right - self._left

    def contains(self, value: float | int) -> bool:
        return self._left <= value <= self._right

    def __contains__(self, value: float | int) -> bool:
        return self.contains(value)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return self._left == other._left and self._right == other._right

    # Lexicographic ordering on (left, right)
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
