from source.numeric_integration.monte_carlo.monte_carlo_numeric_integral import MonteCarloNumericIntegral


class RecursiveMonteCarloIntegral(MonteCarloNumericIntegral):
    def integrate(self) -> float:
        raise NotImplementedError("Recursive MC not implemented")