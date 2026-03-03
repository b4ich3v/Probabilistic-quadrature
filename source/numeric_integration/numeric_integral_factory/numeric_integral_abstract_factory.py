from abc import ABC
from typing import Optional
from abc import abstractmethod

from source.numeric_integration.numeric_integration_pattern import NumericIntegrationPattern
from source.numeric_integration.numeric_integral import NumericIntegral
from source.numeric_integration.monte_carlo.monte_carlo_stretegies import MonteCarloIntegrationStrategy


class NumericIntegralAbstractFactory(ABC):
    @abstractmethod
    def create(self, creation_pattern: NumericIntegrationPattern, monte_carlo_pattern: Optional[MonteCarloIntegrationStrategy], *args, **kwargs) -> NumericIntegral:
        raise RuntimeError("Not implemented yet")
