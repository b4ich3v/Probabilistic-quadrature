from abc import ABC
from abc import abstractmethod

from source.numeric_integration.monte_carlo.monte_carlo_stretegies import MonteCarloIntegrationStrategy
from source.numeric_integration.numeric_integral import NumericIntegral


class MonteCarloAbstractFactory(ABC):
    @abstractmethod
    def create(self, creation_pattern: MonteCarloIntegrationStrategy, *args, **kwargs) -> NumericIntegral:
        raise RuntimeError("Not implemented yet")
