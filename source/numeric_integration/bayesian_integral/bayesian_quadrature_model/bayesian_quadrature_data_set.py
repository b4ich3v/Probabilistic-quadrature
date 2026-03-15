import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass

from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.utils import ensure_2d


# Training data container for BQ: X is (n, d), y is (n,)
@dataclass
class BQDataset:
    X: np.ndarray
    y: np.ndarray

    @classmethod
    def from_arrays(cls, X: ArrayLike, y: ArrayLike) -> "BQDataset":
        X2 = ensure_2d(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if y_arr.shape[0] != X2.shape[0]:
            raise ValueError("X and y must have same first dimension")
        return cls(X2, y_arr)

    # Returns a new dataset with the additional points appended
    def append(self, X_new: ArrayLike, y_new: ArrayLike) -> "BQDataset":
        Xn = ensure_2d(X_new)
        yn = np.asarray(y_new, dtype=float).reshape(-1)
        if yn.shape[0] != Xn.shape[0]:
            raise ValueError("X_new and y_new must have same first dimension")
        X_all = np.vstack([self.X, Xn])
        y_all = np.concatenate([self.y, yn])
        return BQDataset(X_all, y_all)