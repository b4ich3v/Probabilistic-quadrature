import numpy as np

from source.random_variables.continuous_random_variables.uniform import Uniform
from source.numeric_integration.numeric_integral_factory.numeric_integral_factory import NumericIntegralFactory
from source.numeric_integration.numeric_integration_pattern import NumericIntegrationPattern
from source.numeric_integration.monte_carlo.monte_carlo_stretegies import MonteCarloIntegrationStrategy
from source.measures.uniform_box_measure import UniformBoxMeasure
from source.kernels.rbf_kernel import RBFKernel
from source.functions.measured_function import MeasuredFunction
from source.functions.interval import Interval
from source.functions.domain import Domain
from source.functions.function import Function
from source.functions.derivatives.central_difference_derivative_estimator import CentralDifferenceDerivativeEstimator
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.bayesian_quadrature_model import BayesianQuadratureModel
from source.numeric_integration.bayesian_integral.bayesian_quadrature_model.bayesian_quadrature_config import BQConfig


# Integration method factory
factory = NumericIntegralFactory()


def function_predicate(X):
    X = np.atleast_2d(X)
    return np.square(X[:, 0])


def run_and_report(name: str, estimate: float, true_val: float) -> None:
    error = abs(estimate - true_val)
    print(f"{name:<28} estimate={estimate:.6f}  error={error:.2e}")


def uniform_nodes(interval: Interval, sub_intervals: int) -> list[float]:
    step = (interval.get_right_component() - interval.get_left_component()) / sub_intervals
    return [interval.get_left_component() + step * i for i in range(sub_intervals + 1)]


def test_monte_carlo(n_samples: int = 20000, depth: int = 5):
    measure = UniformBoxMeasure(np.array([0.0]), np.array([1.0]))
    mc_function = MeasuredFunction(function_predicate, measure, true_integral=1 / 3)
    rv = Uniform(0.0, 1.0)

    # Standard Monte Carlo
    mc_standard = factory.create(
        NumericIntegrationPattern.MONTE_CARLO,
        MonteCarloIntegrationStrategy.STANDARD,
        mc_function, measure, n_samples=n_samples, rv=rv
    )
    
    # Weighted Monte Carlo (importance sampling)
    proposal = lambda n, r: rv.sample(n)
    weight_fn = lambda x: np.ones(x.shape[0])
    mc_weighted = factory.create(
        NumericIntegrationPattern.MONTE_CARLO,
        MonteCarloIntegrationStrategy.WEIGHTED,
        mc_function, measure, n_samples=n_samples, rv=rv, proposal_sampler=proposal, weight_fn=weight_fn
    )
    
    # Recursive Monte Carlo (1D)
    mc_recursive = factory.create(
        NumericIntegrationPattern.MONTE_CARLO,
        MonteCarloIntegrationStrategy.RECURSIVE,
        mc_function, measure, n_samples=n_samples, rv=rv, depth=depth
    )

    run_and_report("MC standard", mc_standard.integrate(), mc_function.true_integral)
    run_and_report("MC weighted", mc_weighted.integrate(), mc_function.true_integral)
    run_and_report("MC recursive", mc_recursive.integrate(), mc_function.true_integral)


def test_uniform_grid_rules(sub_intervals: int = 256):
    # Rectangle intergration method
    rectangle_integration_method = factory.create(NumericIntegrationPattern.RECTANGLE, None, 
        Function(lambda x: x * x, Domain(Interval(0.0, 1.0)), input_name="x^2"), 
        Interval(0.0, 1.0), uniform_nodes(Interval(0.0, 1.0), sub_intervals), sub_intervals)
    
    # Trapezoid intergration method
    trapezoid_intergration_method = factory.create(NumericIntegrationPattern.TRAPEZOID, None, 
        Function(lambda x: x * x, Domain(Interval(0.0, 1.0)), input_name="x^2"), 
        Interval(0.0, 1.0), uniform_nodes(Interval(0.0, 1.0), sub_intervals), sub_intervals)
    
    # Simpson intergration method
    simpson_intergration_method = factory.create(NumericIntegrationPattern.SIMPSON, None, 
        Function(lambda x: x * x, Domain(Interval(0.0, 1.0)), input_name="x^2"), 
        Interval(0.0, 1.0), uniform_nodes(Interval(0.0, 1.0), sub_intervals), sub_intervals)

    run_and_report("Rectangle", rectangle_integration_method.integrate(), 1.0 / 3.0)
    run_and_report("Trapezoid", trapezoid_intergration_method.integrate(), 1.0 / 3.0)
    run_and_report("Simpson", simpson_intergration_method.integrate(), 1.0 / 3.0)


def test_gaussian_quadratures(n: int = 8):
    # Legendre integration method
    legendre_integration_method = factory.create(NumericIntegrationPattern.LEGENDRE, None, 
        Function(lambda x: 1.0, Domain(Interval(-1.0, 1.0)), input_name="1"), n)
    
    # Hermite integration method
    hermite_integration_method = factory.create(NumericIntegrationPattern.HERMITE, None, 
        Function(lambda x: 1.0, Domain(Interval(float("-inf"), float("inf"))), input_name="1"), n)
    
    # Laguerre integration method
    laguerre_integration_method = factory.create(NumericIntegrationPattern.LAGUERRE, None, 
        Function(lambda x: 1.0, Domain(Interval(0.0, float("inf"))), input_name="1"), n)

    # Chebyshev integration method
    chebyshev_integration_method = factory.create(NumericIntegrationPattern.CHEBYSHEV, None, 
        Function(lambda x: 1.0, Domain(Interval(-1.0, 1.0)), input_name="1"), n)

    run_and_report("Gauss-Legendre", legendre_integration_method.integrate(), 2.0)
    run_and_report("Gauss-Hermite", hermite_integration_method.integrate(), np.sqrt(np.pi))
    run_and_report("Gauss-Laguerre", laguerre_integration_method.integrate(), 1.0)
    run_and_report("Gauss-Chebyshev", chebyshev_integration_method.integrate(), np.pi)


def test_derivative():
    function = Function(lambda x: x ** 2, Domain(Interval(float("-inf"), float("inf"))), input_name="x^2")
    central_der = CentralDifferenceDerivativeEstimator(function, 0.0001)
    central_der.plot_at(2, Interval(-50, 50))


def test_bayesian_quadrature_model(n_points: int = 10, seed: int = 0):
    measure = UniformBoxMeasure(np.array([0.0]), np.array([1.0]))
    true_val = 1.0 / 3.0
    rng = np.random.default_rng(seed)
    X = rng.random((n_points, 1))
    y = function_predicate(X)

    kernel = RBFKernel(lengthscale=0.2, variance=1.0)
    cfg = BQConfig(noise=0.0)
    bq = BayesianQuadratureModel(kernel, measure, config=cfg)
    bq.fit(X, y)
    mean, var = bq.integral_posterior()

    print(f"Bayesian Quadrature        estimate={mean:.6f}  stderr~={np.sqrt(var):.2e}")
    print(f"(true value = {true_val:.6f})")


if __name__ == "__main__":
    test_monte_carlo()
    test_uniform_grid_rules()
    test_gaussian_quadratures()
    test_derivative()
    test_bayesian_quadrature_model()
