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
    
