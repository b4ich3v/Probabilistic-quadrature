from abc import ABC, abstractmethod
import numpy as np


class RandomVariable(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def mean(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def var(self) -> np.ndarray:
        raise NotImplementedError

    def _probability(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_prob(x))

    def calculate_probability(self, x: np.ndarray) -> np.ndarray:
        return self._probability(x)
