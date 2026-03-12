import numpy as np
from numpy.typing import ArrayLike
from typing import Optional
from scipy.linalg import cho_solve

from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.bayesian_quadrature_model import BayesianQuadratureModel
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.utils import kernel_mean_vector, ensure_2d


class ActiveBQSelector:
    def __init__(self, model: BayesianQuadratureModel):
        self.model = model

    def variance_reduction(self, candidate: ArrayLike, noise: Optional[float] = None) -> tuple[float, float]:
        _, var_F = self.model.integral_posterior()
        candidate = ensure_2d(candidate)
        if candidate.shape[0] != 1:
            raise ValueError("candidate must be a single point")

        noise = self.model.config.noise if noise is None else float(noise)

        X = self.model.dataset.X
        mu_f = self.model.mu_f
        L_factor = self.model.L_factor

        k_vec = self.model.kernel(X, candidate)[:, 0]
        kxx = float(self.model.kernel(candidate, candidate)[0, 0] + noise ** 2)
        mu_c = np.asarray(
            kernel_mean_vector(candidate, self.model.kernel, self.model.measure,
                               mc_samples=self.model.config.mc_samples_mean)).squeeze()

        v = cho_solve(L_factor, k_vec)
        s = kxx - k_vec @ v
        if s <= 0:
            return var_F, 0.0

        delta = mu_c - k_vec @ cho_solve(L_factor, mu_f)
        reduction = float((delta ** 2) / s)
        new_var = max(var_F - reduction, 0.0)
        return new_var, reduction

    def greedy_select(self, candidates: ArrayLike, noise: Optional[float] = None) -> tuple[int, float, float]:
        candidates = ensure_2d(candidates)
        best_idx = -1
        best_reduction = -np.inf
        best_new_var = np.inf

        for idx, cand in enumerate(candidates):
            new_var, reduction = self.variance_reduction(cand, noise=noise)
            if reduction > best_reduction:
                best_idx = idx
                best_reduction = reduction
                best_new_var = new_var
        return best_idx, best_new_var, best_reduction


def variance_reduction_with_model(model: BayesianQuadratureModel, candidate: ArrayLike, noise: Optional[float] = None) -> tuple[float, float]:
    return ActiveBQSelector(model).variance_reduction(candidate, noise=noise)


def greedy_select_with_model(model: BayesianQuadratureModel, candidates: ArrayLike, noise: Optional[float] = None) -> tuple[int, float, float]:
    return ActiveBQSelector(model).greedy_select(candidates, noise=noise)
