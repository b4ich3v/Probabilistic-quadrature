from abc import ABC
from abc import abstractmethod

from source.numeric_integration.numeric_integration_pattern import NumericIntegrationPattern
from source.numeric_integration.numeric_integral import NumericIntegral


class NumericIntegralAbstractFactory(ABC):
    @abstractmethod
    def create(self, creation_pattern: NumericIntegrationPattern, *args, **kwargs) -> NumericIntegral:
        raise RuntimeError("Not implemented yet")
