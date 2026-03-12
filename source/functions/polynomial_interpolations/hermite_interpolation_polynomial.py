from source.functions.derivatives.derivative_estimator import DerivativeEstimator
from source.functions.polynomial_interpolations.interpolation_polynomial import InterpolationPoly


class HermiteInterpolationPoly(InterpolationPoly):
    def __init__(self, nodes: list[float], values: list[float], derivative_estimator: DerivativeEstimator) -> None:
        super().__init__(nodes, values)
        if not isinstance(derivative_estimator, DerivativeEstimator):
            raise TypeError("derivative_estimator must be a DerivativeEstimator instance")

        self._derivatives = [derivative_estimator.calculate_derivative_at(x) for x in self._nodes]
        self._z, self._coefficients = self._build_coefficients()

    @property
    def degree(self) -> int:
        return 2 * len(self._nodes) - 1

    def _build_coefficients(self) -> tuple[list[float], list[float]]:
        n = len(self._nodes)
        m = 2 * n
        z = [0.0] * m
        q = [[0.0 for _ in range(m)] for _ in range(m)]

        for idx, x in enumerate(self._nodes):
            z[2 * idx] = z[2 * idx + 1] = x
            q[2 * idx][0] = q[2 * idx + 1][0] = self._values[idx]

        for j in range(1, m):
            for i in range(m - j):
                if j == 1 and z[i] == z[i + 1]:
                    q[i][j] = self._derivatives[i // 2]
                else:
                    denominator = z[i + j] - z[i]
                    if denominator == 0.0:
                        raise ValueError(f"Division by zero in Hermite divided differences at indices {i} and {i + j}")
                    q[i][j] = (q[i + 1][j - 1] - q[i][j - 1]) / denominator

        coefficients = [q[0][j] for j in range(m)]
        return z, coefficients

    def evaluate(self, x: float) -> float:
        result = self._coefficients[-1]
        for i in range(len(self._coefficients) - 2, -1, -1):
            result = self._coefficients[i] + (x - self._z[i]) * result
        return result