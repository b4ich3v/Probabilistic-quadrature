from source.functions.interpolations.interpolation_polynomial import InterpolationPoly


class NewtonInterpolationPoly(InterpolationPoly):
    def __init__(self, nodes: list[float], values: list[float]) -> None:
        super().__init__(nodes, values)

    def _diff(self, nodes, values):
        if len(nodes) == 1:
            return values[0]
        else:
            return (self._diff(nodes[1:], values[1:]) - self._diff(nodes[:-1], values[:-1])) / (nodes[-1] - nodes[0])

    def evaluate(self, x: float) -> float:
            result = 0
            product = 1
            for i in range(len(self._nodes)):
                result += self._diff(self._nodes[:i+1], self._values[:i+1]) * product
                product *= (x - self._nodes[i])
            return result
