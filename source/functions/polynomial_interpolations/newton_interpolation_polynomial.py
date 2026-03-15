from source.functions.polynomial_interpolations.interpolation_polynomial import InterpolationPoly


# Newton form using divided differences, O(n) evaluation via Horner's scheme
class NewtonInterpolationPoly(InterpolationPoly):
    def __init__(self, nodes: list[float], values: list[float]) -> None:
        super().__init__(nodes, values)
        self._coefficients = self._compute_divided_differences()

    # In-place bottom-up divided difference table
    def _compute_divided_differences(self) -> list[float]:
        n = len(self._nodes)
        coeffs = list(self._values)
        for order in range(1, n):
            for i in range(n - 1, order - 1, -1):
                numerator = coeffs[i] - coeffs[i - 1]
                denominator = self._nodes[i] - self._nodes[i - order]
                coeffs[i] = numerator / denominator
        return coeffs

    # Horner's method evaluated right-to-left
    def evaluate(self, x: float) -> float:
        result = self._coefficients[-1]
        for i in range(len(self._nodes) - 2, -1, -1):
            result = self._coefficients[i] + (x - self._nodes[i]) * result
        return result
