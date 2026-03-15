from source.functions.polynomial_interpolations.interpolation_polynomial import InterpolationPoly


# L(x) = sum_i y_i * prod_{j!=i} (x - x_j)/(x_i - x_j), O(n^2) per evaluation
class LagrangeInterpolationPoly(InterpolationPoly):
    def evaluate(self, x: float) -> float:
        result = 0.0
        n = len(self._nodes)
        for i in range(n):
            # Lagrange basis polynomial l_i(x)
            term = self._values[i]
            for j in range(n):
                if i != j:
                    term *= (x - self._nodes[j]) / (self._nodes[i] - self._nodes[j])
            result += term
        return result