from dataclasses import dataclass
import numpy as np


@dataclass
class BQPosteriorState:
    L_factor: tuple
    mu_f: np.ndarray
    sigma_f2: float
