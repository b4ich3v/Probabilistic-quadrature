class Range:
    def __init__(self, input_lower: float, input_upper: float):
        if input_lower > input_upper:
            raise RuntimeError("Range lower bound must not exceed upper bound")
        self.__lower = float(input_lower)
        self.__upper = float(input_upper)

    @property
    def lower(self) -> float:
        return self.__lower

    @property
    def upper(self) -> float:
        return self.__upper

    @property
    def width(self) -> float:
        return self.__upper - self.__lower

    @property
    def middle(self) -> float:
        return 0.5 * (self.__lower + self.__upper)

    def contains(self, y: float) -> bool:
        return self.__lower <= y <= self.__upper

    def __contains__(self, y: float) -> bool:
        return self.contains(y)

    def clamp(self, y: float) -> float:
        return self.__upper if y > self.__upper else self.__lower if y < self.__lower else y

    def __eq__(self, other) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return self.__lower == other.__lower and self.__upper == other.__upper

    def __lt__(self, other) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return (self.__lower, self.__upper) < (other.__lower, other.__upper)

    def __le__(self, other) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return (self.__lower, self.__upper) <= (other.__lower, other.__upper)

    def __gt__(self, other) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return (self.__lower, self.__upper) > (other.__lower, other.__upper)

    def __ge__(self, other) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return (self.__lower, self.__upper) >= (other.__lower, other.__upper)

    def __hash__(self) -> int:
        return hash((self.__lower, self.__upper))

    def __repr__(self) -> str:
        return f"Range({self.__lower}, {self.__upper})"

    def __str__(self) -> str:
        return f"[{self.__lower}, {self.__upper}]"
