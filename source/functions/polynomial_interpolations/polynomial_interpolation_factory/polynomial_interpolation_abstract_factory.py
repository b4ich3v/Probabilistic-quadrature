from abc import ABC
from abc import abstractmethod

from source.functions.polynomial_interpolations.interpolation_polynomial import InterpolationPoly
from source.functions.polynomial_interpolations.interpolation_pattern import InterpolationPattern


class PolyInterpolationAbstractFactory(ABC):
    @abstractmethod
    def create(self, creation_pattern: InterpolationPattern, nodes: list[float], values: list[float]) -> InterpolationPoly:
        raise RuntimeError("Not implemented yet")