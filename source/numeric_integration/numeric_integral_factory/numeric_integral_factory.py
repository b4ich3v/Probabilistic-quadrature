from typing import Optional

from source.numeric_integration.numeric_integration_pattern import NumericIntegrationPattern
from source.numeric_integration.monte_carlo.monte_carlo_factory.monte_carlo_factory import MonteCarloFactory
from source.numeric_integration.monte_carlo.monte_carlo_strategies import MonteCarloIntegrationStrategy
from source.numeric_integration.numeric_integral import NumericIntegral
from source.numeric_integration.uniform_grid_numeric_integrals.rectangle_numeric_integral import RectangleNumericIntegral
from source.numeric_integration.uniform_grid_numeric_integrals.trapezoid_numeric_integral import TrapezoidNumericIntegral
from source.numeric_integration.uniform_grid_numeric_integrals.simpson_numeric_integral import SimpsonNumericIntegral
from source.numeric_integration.weighted_nodes_numeric_integrals.gauss_legendre_integral import GaussLegendreIntegral
from source.numeric_integration.weighted_nodes_numeric_integrals.gauss_hermite_integral import GaussHermiteIntegral
from source.numeric_integration.weighted_nodes_numeric_integrals.gauss_laguerre_integral import GaussLaguerreIntegral
from source.numeric_integration.weighted_nodes_numeric_integrals.gauss_chebyshev_integral import GaussChebyshevIntegral
from source.numeric_integration.bayesian_integral.bayesian_quadrature_integral import BayesianQuadratureIntegral


class NumericIntegralFactory:
    @staticmethod
    def create(pattern: NumericIntegrationPattern, monte_carlo_strategy: Optional[MonteCarloIntegrationStrategy] = None, *args, **kwargs) -> NumericIntegral:
        if pattern == NumericIntegrationPattern.RECTANGLE:
            return RectangleNumericIntegral(*args, **kwargs)
        if pattern == NumericIntegrationPattern.TRAPEZOID:
            return TrapezoidNumericIntegral(*args, **kwargs)
        if pattern == NumericIntegrationPattern.SIMPSON:
            return SimpsonNumericIntegral(*args, **kwargs)
        if pattern == NumericIntegrationPattern.MONTE_CARLO:
            if monte_carlo_strategy is None:
                raise ValueError("monte_carlo_strategy is required for MONTE_CARLO pattern")
            return MonteCarloFactory.create(monte_carlo_strategy, *args, **kwargs)
        if pattern == NumericIntegrationPattern.LEGENDRE:
            return GaussLegendreIntegral(*args, **kwargs)
        if pattern == NumericIntegrationPattern.HERMITE:
            return GaussHermiteIntegral(*args, **kwargs)
        if pattern == NumericIntegrationPattern.LAGUERRE:
            return GaussLaguerreIntegral(*args, **kwargs)
        if pattern == NumericIntegrationPattern.CHEBYSHEV:
            return GaussChebyshevIntegral(*args, **kwargs)
        if pattern == NumericIntegrationPattern.BAYESIAN:
            return BayesianQuadratureIntegral(*args, **kwargs)
        raise ValueError(f"Unsupported integration pattern: {pattern}")
