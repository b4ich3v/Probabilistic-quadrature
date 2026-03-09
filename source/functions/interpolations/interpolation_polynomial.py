from abc import ABC
from abc import abstractmethod


class InterpolationPoly(ABC):
    def __init__(self, nodes: list[float], values: list[float]) -> None:
        if len(nodes) != len(values):
            raise ValueError("nodes and values must be with equal size")
        if len(set(nodes)) != len(nodes):
            raise ValueError("nodes must not contains duplicates")
        if len(nodes) == 0:
            raise ValueError("must be atleast one point")

        self._nodes = list(nodes)
        self._values = list(values)

    @property
    def degree(self) -> int:
        return len(self._nodes) - 1

    @abstractmethod
    def evaluate(self, x: float) -> float:
        raise NotImplementedError