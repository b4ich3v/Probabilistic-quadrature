from typing import Callable, Optional, Sequence

from source.data_structures.domain import Domain
from source.data_structures.range import Range


class Function:
    def __init__(self, input_function: Callable[[float], float], input_domain: Domain, input_codomain: Optional[Range] = None, input_name: str = "f") -> None:
        self.__function = input_function
        self.__domain = input_domain
        self.__range = input_codomain
        self.__name = input_name

    @property
    def domain(self) -> Domain:
        return self.__domain

    @property
    def codomain(self) -> Optional[Range]:
        return self.__range

    @property
    def name(self) -> str:
        return self.__name

    def __call__(self, x: float | Sequence[float]) -> float | list[float]:
        if isinstance(x, (list, tuple)):
            return [self.__evaluate_single(xi) for xi in x]
        return self.__evaluate_single(x)

    def __evaluate_single(self, x: float) -> float:
        if not self.__domain.contains(x):
            bounds = ", ".join(
                f"[{it.get_left_component()}, {it.get_right_component()}]"
                for it in self.__domain.intervals
            )
            raise RuntimeError(f"Input {x} outside domain {bounds}")
        function_result = float(self.__function(x))
        if self.__range and not self.__range.contains(function_result):
            raise RuntimeError(f"Output {function_result} outside declared range [{self.__range.lower}, {self.__range.upper}]")
        return function_result
