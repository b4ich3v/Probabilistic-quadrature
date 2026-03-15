from source.functions.derivatives.derivative_estimator import DerivativeEstimator


# Forward difference: (f(x+h) - f(x)) / h, O(h) accuracy
class ForwardDerivativeEstimator(DerivativeEstimator):
    def calculate_derivative_at(self, point: float) -> float:
        return (self._function(point + self._precision) - self._function(point)) / self._precision