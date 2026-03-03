import numpy as np

from source.numeric_integration.numeric_integral_factory.numeric_integral_factory import NumericIntegralFactory
from source.numeric_integration.numeric_integration_pattern import NumericIntegrationPattern
from source.numeric_integration.monte_carlo.monte_carlo_stretegies import MonteCarloIntegrationStrategy
from source.measures.uniform_box_measure import UniformBoxMeasure
from source.functions.measured_function import MeasuredFunction


def function_predicate(X):
    X = np.atleast_2d(X)
    return np.square(X[:, 0])


factory = NumericIntegralFactory()
measure = UniformBoxMeasure(np.array([0.0]), np.array([1.0]))
input_function = MeasuredFunction(function_predicate, measure, true_integral = 1 / 3)


def test1():
    input_function = MeasuredFunction(function_predicate, measure, true_integral = 1 / 3)
    integral_1 = factory.create(
        NumericIntegrationPattern.MONTE_CARLO,
        MonteCarloIntegrationStrategy.STANDARD,
        input_function, measure, n_samples=20000, rng=np.random.default_rng(0)
    )

    print("MC standard:", integral_1.integrate())


def test2():
    input_function = MeasuredFunction(function_predicate, measure, true_integral = 1 / 3)
    proposal = lambda n, rng: rng.random((n, 1))
    weight_fn = lambda x: np.ones(x.shape[0])

    integral_w = factory.create(
        NumericIntegrationPattern.MONTE_CARLO,
        MonteCarloIntegrationStrategy.WEIGHTED,
        input_function, measure, n_samples=20000, rng=np.random.default_rng(0), proposal_sampler=proposal, weight_fn=weight_fn
    )

    print("MC weighted:", integral_w.integrate())


def test3():
    integral_r = factory.create(
        NumericIntegrationPattern.MONTE_CARLO,
        MonteCarloIntegrationStrategy.RECURSIVE,
        input_function, measure, n_samples=20000, rng=np.random.default_rng(0), depth=5
    )

    print("MC recursive:", integral_r.integrate())


if __name__ == "__main__":
    test1()
    test2()
    test3()