import math

from source.functions.polynomial_interpolations.interpolation_polynomial import InterpolationPoly


class HermitInterpolationPoly(InterpolationPoly):
    def __init__(self, nodes: list[float], values: list[float]) -> None:
        super().__init__(nodes, values)

    def _diff(self, left: int, right: int):
        if self._nodes[left] == self._nodes[right]: 
            return self._values[left] / math.factorial(right - left)
        return (self._diff(self._nodes, self._values, left + 1, right) - 
                self._diff(self._nodes, self._values, left, right - 1)) / (
                self._nodes[right] - self._nodes[left])

    def evaluate(self, x: float) -> float:
        result = 0
        product = 1
        for i in range(len(self._values)):
            result += self._diff(self._nodes, self._values, 0, i) * product
            product *= (x - self._nodes[i])
        return result