from source.functions.derivatives.derivative_estimator import DerivativeEstimator
from source.functions.function import Function


class ForwardDerivativeEstimator(DerivativeEstimator):
    def __init__(self, function: Function, precision: float) -> None:
        super().__init__(function, precision)

    def calculate_derivative_at(self, point: float) -> float:
        return (self._function(point + self._precision) - self._function(point)) / self._precision