import numpy as np
from typing import Optional

from source.numeric_integration.numeric_integral import NumericIntegral
from source.functions.interval import Interval
from source.functions.function import Function
from source.measures.measure import Measure
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.bayesian_quadrature_model import BayesianQuadratureModel
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.bayesian_quadrature_config import BQConfig


class BayesianQuadratureIntegral(NumericIntegral):
    def __init__(self, input_function: Function, measure: Measure, kernel,
        X: np.ndarray, y: np.ndarray, noise: float = 0.0, jitter: float = 1e-8) -> None:
        super().__init__(input_function, Interval(0.0, 1.0), [0.0, 1.0], 1, validate=False)
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
