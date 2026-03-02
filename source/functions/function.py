from typing import Callable, Optional, Sequence, Union, List

from source.functions.domain import Domain
from source.functions.range import Range


class Function:
    def __init__(self, input_function: Callable[[float], float], input_domain: Domain, input_codomain: Optional[Range] = None, input_name: str = "f") -> None:
        self._function = input_function
        self._domain = input_domain
        self._range = input_codomain
        self._name = input_name

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def codomain(self) -> Optional[Range]:
        return self._range

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, input_values: Union[float, Sequence[float]]) -> Union[float, List[float]]:
        if isinstance(input_values, (list, tuple)):
            return [self._evaluate_single(current_input_value) for current_input_value in input_values]
        return self._evaluate_single(input_values)

    def _evaluate_single(self, x: float) -> float:
        if not self._domain.contains(x):
            bounds = ", ".join(
                f"[{it.get_left_component()}, {it.get_right_component()}]"
                for it in self._domain.intervals
            )
            raise RuntimeError(f"Input {x} outside domain {bounds}")
        function_result = float(self._function(x))
        if self._range and not self._range.contains(function_result):
            raise RuntimeError(f"Output {function_result} outside declared range [{self._range.lower}, {self._range.upper}]")
        return function_result
