from source.numeric_integration.monte_carlo.monte_carlo_stretegies import MonteCarloIntegrationStrategy
from source.numeric_integration.monte_carlo.monte_carlo_numeric_integral import MonteCarloNumericIntegral
from source.numeric_integration.monte_carlo.monte_carlo_factory.monte_carlo_abstract_factory import MonteCarloAbstractFactory
from source.numeric_integration.monte_carlo.standard_monte_carlo_integral import StandardMonteCarloIntegral
from source.numeric_integration.monte_carlo.recursive_monte_carlo_integral import RecursiveMonteCarloIntegral
from source.numeric_integration.monte_carlo.weighted_monte_carlo_integral import WeightedMonteCarloIntegral


class MonteCarloFactory(MonteCarloAbstractFactory):
    def create(self, creation_pattern: MonteCarloIntegrationStrategy, *args, **kwargs) -> MonteCarloNumericIntegral:
        if creation_pattern == MonteCarloIntegrationStrategy.STANDARD: return StandardMonteCarloIntegral(*args, **kwargs)
        if creation_pattern == MonteCarloIntegrationStrategy.RECURSIVE: return RecursiveMonteCarloIntegral(*args, **kwargs)
        if creation_pattern == MonteCarloIntegrationStrategy.WEIGHTED: return WeightedMonteCarloIntegral(*args, **kwargs)
        raise ValueError(f"Unsupported integration pattern: {creation_pattern}")
