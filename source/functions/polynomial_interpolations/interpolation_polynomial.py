from abc import ABC, abstractmethod


# Base class for polynomial interpolation through (node, value) pairs
class InterpolationPoly(ABC):
    def __init__(self, nodes: list[float], values: list[float]) -> None:
        if len(nodes) == 0:
            raise ValueError("Must have at least one interpolation point")
        if len(nodes) != len(values):
            raise ValueError("nodes and values must have equal size")
        if len(set(nodes)) != len(nodes):
            raise ValueError("nodes must not contain duplicates")

        self._nodes = list(nodes)
        self._values = list(values)

    @property
    def degree(self) -> int:
        return len(self._nodes) - 1

    @abstractmethod
    def evaluate(self, x: float) -> float: ...

    def __call__(self, x: float) -> float:
        return self.evaluate(x)