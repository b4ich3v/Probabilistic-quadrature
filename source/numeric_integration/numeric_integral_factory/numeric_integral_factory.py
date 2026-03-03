from typing import Optional

from source.numeric_integration.numeric_integration_pattern import NumericIntegrationPattern
from source.numeric_integration.monte_carlo.monte_carlo_factory.monte_carlo_factory import MonteCarloFactory
from source.numeric_integration.monte_carlo.monte_carlo_stretegies import MonteCarloIntegrationStrategy
from source.numeric_integration.numeric_integral import NumericIntegral
from source.numeric_integration.uniform_grid_numeric_integrals.rectangle_numeric_integral import RectangleNumericIntegral
from source.numeric_integration.uniform_grid_numeric_integrals.trapezoid_numeric_integral import TrapezoidNumericIntegral
from source.numeric_integration.uniform_grid_numeric_integrals.simpson_numeric_intergral import SimpsonNumericIntegral
from source.numeric_integration.weighted_nodes_numeric_integrals.gauss_legendre_integral import GaussLegendreIntegral
from source.numeric_integration.weighted_nodes_numeric_integrals.gauss_hermite_integral import GaussHermiteIntegral
from source.numeric_integration.weighted_nodes_numeric_integrals.gauss_laguerre_integral import GaussLaguerreIntegral
from source.numeric_integration.weighted_nodes_numeric_integrals.gauss_chebyshev_integral import GaussChebyshevIntegral
from source.numeric_integration.numeric_integral_factory.numeric_integral_abstract_factory import NumericIntegralAbstractFactory


class NumericIntegralFactory(NumericIntegralAbstractFactory):
    def create(self, creation_pattern: NumericIntegrationPattern, monte_carlo_pattern: Optional[MonteCarloIntegrationStrategy], *args, **kwargs) -> NumericIntegral:
        if creation_pattern == NumericIntegrationPattern.RECTANGLE: return RectangleNumericIntegral(*args, **kwargs)
        if creation_pattern == NumericIntegrationPattern.TRAPEZOID: return TrapezoidNumericIntegral(*args, **kwargs)
        if creation_pattern == NumericIntegrationPattern.SIMPSON: return SimpsonNumericIntegral(*args, **kwargs)
        if creation_pattern == NumericIntegrationPattern.MONTE_CARLO: return MonteCarloFactory().create(monte_carlo_pattern, *args, **kwargs)
        if creation_pattern == NumericIntegrationPattern.LEGENDRE: return GaussLegendreIntegral(*args, **kwargs)
        if creation_pattern == NumericIntegrationPattern.HERMITE: return GaussHermiteIntegral(*args, **kwargs)
        if creation_pattern == NumericIntegrationPattern.LAGUERRE: return GaussLaguerreIntegral(*args, **kwargs)
        if creation_pattern == NumericIntegrationPattern.CHEBYSHEV: return GaussChebyshevIntegral(*args, **kwargs)
        raise ValueError(f"Unsupported integration pattern: {creation_pattern}")
