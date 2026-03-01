class Interval:
    def __init__(self, left_component: float | int, right_component: float | int) -> None:
        if left_component > right_component:
            raise RuntimeError("Right component must be greater than left")
        if not isinstance(left_component, (int, float)) or not isinstance(right_component, (int, float)):
            raise RuntimeError("Left and right input components must be numeric")
        self.__set_left_component(float(left_component))
        self.__set_right_component(float(right_component))

    def __set_left_component(self, left) -> None:
        self.__left = left

    def __set_right_component(self, right) -> None:
        self.__right = right

    def get_left_component(self) -> float | int:
        return self.__left

    def get_right_component(self) -> float | int:
        return self.__right

    def get_len_of_interval(self) -> float:
        return self.__right - self.__left

    @property
    def width(self) -> float:
        return self.__right - self.__left

    def contains(self, input_value: int | float) -> bool:
        return self.__left <= input_value and input_value <= self.__right

    def __contains__(self, input_value: int | float) -> bool:
        return self.contains(input_value)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return self.__left == other.__left and self.__right == other.__right

    def __lt__(self, other) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return (self.__left, self.__right) < (other.__left, other.__right)

    def __le__(self, other) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return (self.__left, self.__right) <= (other.__left, other.__right)

    def __gt__(self, other) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return (self.__left, self.__right) > (other.__left, other.__right)

    def __ge__(self, other) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return (self.__left, self.__right) >= (other.__left, other.__right)

    def __hash__(self) -> int:
        return hash((self.__left, self.__right))

    def __repr__(self) -> str:
        return f"Interval({self.__left}, {self.__right})"

    def __str__(self) -> str:
        return f"[{self.__left}, {self.__right}]"
