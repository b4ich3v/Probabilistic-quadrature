import numpy as np

from source.functions.interval import Interval


# Maps [-1, 1] reference nodes and weights to an arbitrary [a, b] interval
class AffineTransformation:
    @staticmethod
    def map_from_unit(nodes: np.ndarray, weights: np.ndarray, interval: Interval) -> tuple[np.ndarray, np.ndarray]:
        a = interval.left
        b = interval.right
        scale = 0.5 * (b - a)
        shift = 0.5 * (a + b)
        mapped_nodes = scale * nodes + shift
        mapped_weights = scale * weights
        return mapped_nodes, mapped_weights