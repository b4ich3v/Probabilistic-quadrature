from source.functions.derivatives.derivative_estimator import DerivativeEstimator


# Central difference: (f(x+h) - f(x-h)) / 2h, O(h^2) accuracy
class CentralDifferenceDerivativeEstimator(DerivativeEstimator):
    def calculate_derivative_at(self, point: float) -> float:
        h = self._precision
        return (self._function(point + h) - self._function(point - h)) / (2 * h)