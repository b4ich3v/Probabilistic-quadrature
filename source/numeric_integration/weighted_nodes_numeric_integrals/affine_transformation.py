import numpy as np

from source.functions.interval import Interval


class AffineTransformation:
    @staticmethod
    def map_from_unit(nodes: np.ndarray, weights: np.ndarray, interval: Interval) -> tuple[np.ndarray, np.ndarray]:
        a = interval.get_left_component()
        b = interval.get_right_component()
        scale = 0.5 * (b - a)
        shift = 0.5 * (a + b)
        mapped_nodes = scale * nodes + shift  # x = ((b - a) / 2) * t + (a + b) / 2
        mapped_weights = scale * weights  # w' = ((b - a) / 2) * w
        return mapped_nodes, mapped_weights
