import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple

from source.kernels.kernel import Kernel
from source.numeric_integration.bayesian_integral.gaussian_process import GaussianProcess
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.bayesian_quadrature_config import BQConfig
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.bayesian_quadrature_data_set import BQDataset
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.bayesian_quadrature_integral_terms import BQIntegralTermsComputer
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.bayesian_quadrature_posterior_cache import BQPosteriorState
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.utils import gp_posterior_predictive


class BayesianQuadratureModel:
    def __init__(self, kernel: Kernel, measure, config: Optional[BQConfig] = None):
        self.kernel = kernel
        self.measure = measure
        self.config = config or BQConfig()
        self.config.validate()
        self.dataset: Optional[BQDataset] = None
        self._gp = GaussianProcess(kernel, noise=self.config.noise, jitter=self.config.jitter)
        self._integral_terms = BQIntegralTermsComputer(
            kernel, measure, self.config.mc_samples_mean, self.config.mc_samples_var
        )
        self._state: Optional[BQPosteriorState] = None

    def _check_fitted(self) -> None:
        if self._state is None:
            raise RuntimeError("BayesianQuadratureModel is not fitted yet")

    def fit(self, X: ArrayLike, y: ArrayLike) -> "BayesianQuadratureModel":
        ds = BQDataset.from_arrays(X, y)
        self.dataset = ds
        self._recompute_cache()
        return self

    def update(self, X_new: ArrayLike, y_new: ArrayLike) -> "BayesianQuadratureModel":
        if self.dataset is None:
            return self.fit(X_new, y_new)
        self.dataset = self.dataset.append(X_new, y_new)
        self._recompute_cache()
        return self

    def _recompute_cache(self) -> None:
        ds = self.dataset
        assert ds is not None
        self._gp.fit(ds.X, ds.y)
        mu_f, sigma_f2 = self._integral_terms.compute(ds.X)
        self._state = BQPosteriorState(K_inv=self._gp.K_inv, mu_f=mu_f, sigma_f2=sigma_f2)

    def integral_posterior(self) -> Tuple[float, float]:
        self._check_fitted()
        state = self._state
        mu_f = state.mu_f
        sigma_f2 = state.sigma_f2
        K_inv = state.K_inv
        y_train = self.dataset.y
        mean_F = mu_f @ (K_inv @ y_train)
        var_F = sigma_f2 - mu_f @ (K_inv @ mu_f)
        return float(mean_F), float(np.maximum(var_F, 0.0))

    def predict(self, X_test: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        return gp_posterior_predictive(
            self.dataset.X, self.dataset.y, X_test,
            self.kernel, noise=self.config.noise, K_inv=self._gp.K_inv
        )

    @property
    def K_inv(self) -> Optional[np.ndarray]:
        return None if self._state is None else self._state.K_inv

    @property
    def mu_f(self) -> Optional[np.ndarray]:
        return None if self._state is None else self._state.mu_f

    @property
    def sigma_f2(self) -> Optional[float]:
        return None if self._state is None else self._state.sigma_f2
