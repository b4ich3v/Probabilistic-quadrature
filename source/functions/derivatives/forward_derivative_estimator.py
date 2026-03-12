from source.functions.derivatives.derivative_estimator import DerivativeEstimator


class ForwardDerivativeEstimator(DerivativeEstimator):
    def calculate_derivative_at(self, point: float) -> float:
        return (self._function(point + self._precision) - self._function(point)) / self._precision