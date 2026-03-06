import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from abc import abstractmethod

from source.functions.function import Function
from source.functions.interval import Interval


class DerivativeEstimator(ABC):
    def __init__(self, function: Function, precision: float) -> None:
        self._function = function
        self._precision = precision

    @abstractmethod
    def calculate_derivative_at(self, point: float) -> float:
        raise NotImplementedError
    
    def plot_at(self, point: float, interval: Interval) -> None:
        x_coordinates = np.linspace(interval.get_left_component(), interval.get_right_component(), 1000)
        y_coordinates = []
        y_derivatvie_coordinats = []
        derivative_function = lambda x, point: self._function(point) + self.calculate_derivative_at(point) * (x - point)

        for i in range(0, len(x_coordinates)):
            y_coordinates.append(self._function(x_coordinates[i]))
            y_derivatvie_coordinats.append(derivative_function(x_coordinates[i], point))
        
        plt.plot(x_coordinates, y_coordinates)
        plt.plot(x_coordinates, y_derivatvie_coordinats)
        plt.show()