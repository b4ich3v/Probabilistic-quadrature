from source.functions.interpolations.interpolation_polynomial import InterpolationPoly


class LagrangeInterpolationPoly(InterpolationPoly):
    def __init__(self, nodes: list[float], values: list[float]):
        super().__init__(nodes, values)

    def evaluate(self, x: float) -> float:
        result = 0
        for i in range(len(self._nodes)):
            term = self._values[i]
            for j in range(len(self._nodes)):
                if i != j:
                    term *= (x - self._nodes[j]) / (self._nodes[i] - self._nodes[j])
            result += term
        return result