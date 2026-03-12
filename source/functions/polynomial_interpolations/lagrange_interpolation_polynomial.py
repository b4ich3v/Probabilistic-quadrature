from source.functions.polynomial_interpolations.interpolation_polynomial import InterpolationPoly


class LagrangeInterpolationPoly(InterpolationPoly):
    def evaluate(self, x: float) -> float:
        result = 0.0
        n = len(self._nodes)
        for i in range(n):
            term = self._values[i]
            for j in range(n):
                if i != j:
                    term *= (x - self._nodes[j]) / (self._nodes[i] - self._nodes[j])
            result += term
        return result