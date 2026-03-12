class Range:
    def __init__(self, lower: float, upper: float):
        if lower > upper:
            raise ValueError("Range lower bound must not exceed upper bound")
        self._lower = float(lower)
        self._upper = float(upper)

    @property
    def lower(self) -> float:
        return self._lower

    @property
    def upper(self) -> float:
        return self._upper

    @property
    def width(self) -> float:
        return self._upper - self._lower

    @property
    def middle(self) -> float:
        return 0.5 * (self._lower + self._upper)

    def contains(self, y: float) -> bool:
        return self._lower <= y <= self._upper

    def __contains__(self, y: float) -> bool:
        return self.contains(y)

    def clamp(self, y: float) -> float:
        return self._upper if y > self._upper else self._lower if y < self._lower else y

    def __eq__(self, other) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return self._lower == other._lower and self._upper == other._upper

    def __lt__(self, other) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return (self._lower, self._upper) < (other._lower, other._upper)

    def __le__(self, other) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return (self._lower, self._upper) <= (other._lower, other._upper)

    def __gt__(self, other) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return (self._lower, self._upper) > (other._lower, other._upper)

    def __ge__(self, other) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return (self._lower, self._upper) >= (other._lower, other._upper)

    def __hash__(self) -> int:
        return hash((self._lower, self._upper))

    def __repr__(self) -> str:
        return f"Range({self._lower}, {self._upper})"

    def __str__(self) -> str:
        return f"[{self._lower}, {self._upper}]"
