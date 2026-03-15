import numpy as np
from typing import Optional

from source.numeric_integration.numeric_integral import NumericIntegral
from source.functions.function import Function
from source.measures.measure import Measure
from source.kernels.kernel import Kernel
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.bayesian_quadrature_model import BayesianQuadratureModel
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.bayesian_quadrature_config import BQConfig


# NumericIntegral adapter: fits a BQ model and returns posterior mean as the estimate
class BayesianQuadratureIntegral(NumericIntegral):
    def __init__(self, func, measure: Measure, kernel: Kernel,
                 X: np.ndarray, y: np.ndarray, noise: float = 0.0, jitter: float = 1e-8) -> None:
        self._function = func
        self._variance: Optional[float] = None
        config = BQConfig(noise=noise, jitter=jitter)
        self._model = BayesianQuadratureModel(kernel, measure, config=config).fit(X, y)

    @property
    def variance(self) -> Optional[float]:
        return self._variance

    def integrate(self) -> float:
        mean, var = self._model.integral_posterior()
        self._variance = var
        return mean