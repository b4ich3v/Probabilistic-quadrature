from abc import ABC, abstractmethod
import numpy as np


class RandomVariable(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> np.ndarray: ...

    @abstractmethod
    def log_prob(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def mean(self) -> np.ndarray: ...

    @abstractmethod
    def var(self) -> np.ndarray: ...

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_prob(x))