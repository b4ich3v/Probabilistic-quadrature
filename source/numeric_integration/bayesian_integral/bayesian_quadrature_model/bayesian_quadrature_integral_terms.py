import numpy as np

from source.kernels.kernel import Kernel
from source.measures.measure import Measure
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.utils import kernel_mean_vector, kernel_integral_variance


# Computes mu_f = E_mu[k(X, .)] and sigma_f^2 = E_mu[E_mu[k(., .)]]
class BQIntegralTermsComputer:
    def __init__(self, kernel: Kernel, measure: Measure, mc_samples_mean: int, mc_samples_var: int):
        self.kernel = kernel
        self.measure = measure
        self.mc_samples_mean = mc_samples_mean
        self.mc_samples_var = mc_samples_var

    def compute(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        mu_f = kernel_mean_vector(X, self.kernel, self.measure, mc_samples=self.mc_samples_mean)
        sigma_f2 = kernel_integral_variance(self.kernel, self.measure, mc_samples=self.mc_samples_var)
        return mu_f, float(sigma_f2)