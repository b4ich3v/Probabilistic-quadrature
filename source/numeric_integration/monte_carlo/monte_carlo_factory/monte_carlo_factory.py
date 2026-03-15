from source.numeric_integration.monte_carlo.monte_carlo_strategies import MonteCarloIntegrationStrategy
from source.numeric_integration.monte_carlo.monte_carlo_numeric_integral import MonteCarloNumericIntegral
from source.numeric_integration.monte_carlo.standard_monte_carlo_integral import StandardMonteCarloIntegral
from source.numeric_integration.monte_carlo.recursive_monte_carlo_integral import RecursiveMonteCarloIntegral
from source.numeric_integration.monte_carlo.weighted_monte_carlo_integral import WeightedMonteCarloIntegral


# Dispatches MC integrator construction by strategy enum
class MonteCarloFactory:
    @staticmethod
    def create(pattern: MonteCarloIntegrationStrategy, *args, **kwargs) -> MonteCarloNumericIntegral:
        if pattern == MonteCarloIntegrationStrategy.STANDARD:
            return StandardMonteCarloIntegral(*args, **kwargs)
        if pattern == MonteCarloIntegrationStrategy.RECURSIVE:
            return RecursiveMonteCarloIntegral(*args, **kwargs)
        if pattern == MonteCarloIntegrationStrategy.WEIGHTED:
            return WeightedMonteCarloIntegral(*args, **kwargs)
        raise ValueError(f"Unsupported Monte Carlo strategy: {pattern}")
