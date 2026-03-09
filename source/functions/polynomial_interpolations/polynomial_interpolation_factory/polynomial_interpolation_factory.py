from source.functions.derivatives.central_difference_derivative_estimator import CentralDifferenceDerivativeEstimator
from source.functions.polynomial_interpolations.interpolation_polynomial import InterpolationPoly
from source.functions.polynomial_interpolations.interpolation_pattern import InterpolationPattern
from source.functions.polynomial_interpolations.polynomial_interpolation_factory.polynomial_interpolation_abstract_factory import PolyInterpolationAbstractFactory
from source.functions.polynomial_interpolations.lagrange_interpolation_polynomial import LagrangeInterpolationPoly
from source.functions.polynomial_interpolations.hermit_interpolation_polynomial import HermitInterpolationPoly
from source.functions.polynomial_interpolations.newton_interpolation_polynomial import NewtonInterpolationPoly


class PolynomialInterpolationFactory(PolyInterpolationAbstractFactory):
    def create(self, creation_pattern: InterpolationPattern, nodes: list[float], values: list[float], *args) -> InterpolationPoly:
        if creation_pattern == InterpolationPattern.LAGRANGE: return LagrangeInterpolationPoly(nodes, values)
        if creation_pattern == InterpolationPattern.HERMIT: return HermitInterpolationPoly(nodes, values, CentralDifferenceDerivativeEstimator(*args))
        if creation_pattern == InterpolationPattern.NEWTON: return NewtonInterpolationPoly(nodes, values)
        raise ValueError(f"Unsupported polynomial interpolation pattern: {creation_pattern}")