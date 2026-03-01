from source.numeric_integration.numeric_integration_pattern import NumericIntegrationPattern
from source.numeric_integration.numeric_integral import NumericIntegral
from source.numeric_integration.rectangle_numeric_integral import RectangleNumericIntegral
from source.numeric_integration.trapezoid_numeric_integral import TrapezoidNumericIntegral
from source.numeric_integration.simpson_numeric_intergral import SimpsonNumericIntegral
from source.numeric_integration.monte_carlo_numeric_integral import MonteCarloNumericIntegral
from source.numeric_integration.numeric_integral_factory.numeric_integral_abstract_factory import NumericIntegralAbstractFactory


class NumericIntegralFactory(NumericIntegralAbstractFactory):
    def create(self, creation_pattern: NumericIntegrationPattern, *args, **kwargs) -> NumericIntegral:
        if creation_pattern == NumericIntegrationPattern.RECTANGLE:
            return RectangleNumericIntegral(*args, **kwargs)
        if creation_pattern == NumericIntegrationPattern.TRAPEZOID:
            return TrapezoidNumericIntegral(*args, **kwargs)
        if creation_pattern == NumericIntegrationPattern.SIMPSON:
            return SimpsonNumericIntegral(*args, **kwargs)
        if creation_pattern == NumericIntegrationPattern.MONTE_CARLO:
            return MonteCarloNumericIntegral(*args, **kwargs)
        raise ValueError(f"Unsupported integration pattern: {creation_pattern}")
