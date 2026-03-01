from abc import ABC, abstractmethod
import numpy as np


class Measure(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        raise RuntimeError("Not implemented yet")

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        raise RuntimeError("Not implemented yet")

