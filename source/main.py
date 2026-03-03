import numpy as np

from source.numeric_integration.numeric_integral_factory.numeric_integral_factory import NumericIntegralFactory
from source.numeric_integration.numeric_integration_pattern import NumericIntegrationPattern
from source.numeric_integration.monte_carlo.monte_carlo_stretegies import MonteCarloIntegrationStrategy
from source.measures.uniform_box_measure import UniformBoxMeasure


def test1():
    factory = NumericIntegralFactory()
    measure = UniformBoxMeasure(np.array([0.0]), np.array([1.0]))

    def function_predicate(X):
        X = np.atleast_2d(X)
        return np.square(X[:, 0])

    factory = NumericIntegralFactory()
    integral_1 = factory.create(
        NumericIntegrationPattern.MONTE_CARLO, 
        MonteCarloIntegrationStrategy.STANDARD, 
        function_predicate, measure, 20000, np.random.default_rng(0))

    print(integral_1.integrate())


if __name__ == "__main__":
    test1()