from enum import Enum


class NumericIntegrationPattern(Enum):
    RECTANGLE = 0
    TRAPEZOID = 1
    SIMPSON = 2
    MONTE_CARLO = 3
    LEGENDRE = 4
    HERMITE = 5
    LAGUERRE = 6
    CHEBYSHEV = 7
