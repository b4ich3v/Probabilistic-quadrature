from abc import ABC, abstractmethod
import numpy as np


class Measure(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray: ...
