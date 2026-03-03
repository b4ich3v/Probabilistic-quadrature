from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from source.random_variables.random_variable import RandomVariable


class Measure(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        raise RuntimeError("Not implemented yet")

    @abstractmethod
    def sample(self, n: int, rv: Optional[RandomVariable] = None) -> np.ndarray:
        raise RuntimeError("Not implemented yet")
