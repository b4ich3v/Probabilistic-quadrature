from typing import Optional

from source.functions.derivatives.derivative_estimator import DerivativeEstimator
from source.functions.derivatives.central_difference_derivative_estimator import CentralDifferenceDerivativeEstimator
from source.functions.polynomial_interpolations.interpolation_polynomial import InterpolationPoly
from source.functions.polynomial_interpolations.interpolation_pattern import InterpolationPattern
from source.functions.polynomial_interpolations.lagrange_interpolation_polynomial import LagrangeInterpolationPoly
from source.functions.polynomial_interpolations.hermite_interpolation_polynomial import HermiteInterpolationPoly
from source.functions.polynomial_interpolations.newton_interpolation_polynomial import NewtonInterpolationPoly


class PolynomialInterpolationFactory:
    @staticmethod
    def create(pattern: InterpolationPattern, nodes: list[float], values: list[float], *args, derivative_estimator: Optional[DerivativeEstimator] = None) -> InterpolationPoly:
        if pattern == InterpolationPattern.LAGRANGE:
            return LagrangeInterpolationPoly(nodes, values)
        if pattern == InterpolationPattern.NEWTON:
            return NewtonInterpolationPoly(nodes, values)
        if pattern == InterpolationPattern.HERMITE:
            if derivative_estimator is None:
                if len(args) >= 2:
                    derivative_estimator = CentralDifferenceDerivativeEstimator(args[0], args[1])
                elif len(args) == 1 and isinstance(args[0], DerivativeEstimator):
                    derivative_estimator = args[0]
                else:
                    raise ValueError("Hermite interpolation requires a DerivativeEstimator or (Function, precision) args")
            return HermiteInterpolationPoly(nodes, values, derivative_estimator)
        raise ValueError(f"Unsupported interpolation pattern: {pattern}")