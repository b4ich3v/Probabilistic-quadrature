from typing import Callable, Optional, Sequence
import numpy as np

from source.functions.domain import Domain
from source.functions.range import Range


# Callable wrapper with domain and optional codomain validation
class Function:
    def __init__(self, func: Callable[[float], float], domain: Domain, codomain: Optional[Range] = None, name: str = "f") -> None:
        self._function = func
        self._domain = domain
        self._range = codomain
        self._name = name

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def codomain(self) -> Optional[Range]:
        return self._range

    @property
    def name(self) -> str:
        return self._name

    # Dispatch by input type: scalar returns scalar, collection returns list
    def __call__(self, input_values: float | Sequence[float] | np.ndarray) -> float | list[float]:
        if isinstance(input_values, np.ndarray):
            return [self._evaluate_single(float(x)) for x in input_values.flat]
        if isinstance(input_values, (list, tuple)):
            return [self._evaluate_single(x) for x in input_values]
        return self._evaluate_single(input_values)

    def _evaluate_single(self, x: float) -> float:
        if not self._domain.contains(x):
            bounds = ", ".join(
                f"[{iv.left}, {iv.right}]"
                for iv in self._domain.intervals
            )
            raise ValueError(f"Input {x} outside domain {bounds}")
        result = float(self._function(x))
        if self._range and not self._range.contains(result):
            raise ValueError(f"Output {result} outside declared range [{self._range.lower}, {self._range.upper}]")
        return result
