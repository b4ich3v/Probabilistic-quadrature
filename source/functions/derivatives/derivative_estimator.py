import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from source.functions.function import Function
from source.functions.interval import Interval


class DerivativeEstimator(ABC):
    def __init__(self, function: Function, precision: float) -> None:
        if precision <= 0:
            raise ValueError("precision must be positive")
        self._function = function
        self._precision = precision

    @abstractmethod
    def calculate_derivative_at(self, point: float) -> float: ...

    def plot_at(self, point: float, interval: Interval) -> None:
        x_coords = np.linspace(interval.left, interval.right, 1000)
        f_at_point = self._function(point)
        d_at_point = self.calculate_derivative_at(point)

        y_coords = np.array([self._function(x) for x in x_coords])
        y_tangent = f_at_point + d_at_point * (x_coords - point)

        plt.plot(x_coords, y_coords)
        plt.plot(x_coords, y_tangent)
        plt.show()