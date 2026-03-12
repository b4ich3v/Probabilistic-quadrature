from abc import abstractmethod, ABC


class NumericIntegral(ABC):
    @abstractmethod
    def integrate(self) -> float: ...
