from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Measure(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        raise RuntimeError("Not implemented yet")

    @abstractmethod
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        raise RuntimeError("Not implemented yet")
