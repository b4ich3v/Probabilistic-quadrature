from dataclasses import dataclass
import numpy as np


@dataclass
class BQPosteriorState:
    K_inv: np.ndarray
    mu_f: np.ndarray
    sigma_f2: float
