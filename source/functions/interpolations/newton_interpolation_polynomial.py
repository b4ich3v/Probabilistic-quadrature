from source.functions.interpolations.interpolation_polynomial import InterpolationPoly


class NewtonInterpolationPoly(InterpolationPoly):
    def __init__(self, nodes: list[float], values: list[float]) -> None:
        super().__init__(nodes, values)

    def _diff(self):
        if len(self._nodes) == 1:
            return self._values[0]
        else:
            return (self._diff(self._nodes[1:], self._values[1:]) - 
                    self._diff(self._nodes[:-1], self._values[:-1])) / (
                    self._nodes[-1] - self._nodes[0])

    def evaluate(self, x: float) -> float:
            result = 0
            product = 1
            for i in range(len(self._nodes)):
                result += self._diff(self._nodes[:i+1], self._values[:i+1]) * product
                product *= (x - self._nodes[i])
            return result
