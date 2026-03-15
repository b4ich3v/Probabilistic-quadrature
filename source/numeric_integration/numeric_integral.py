from abc import abstractmethod, ABC


# Single-method interface shared by all integration backends
class NumericIntegral(ABC):
    @abstractmethod
    def integrate(self) -> float: ...
