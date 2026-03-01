from enum import Enum


class NumericIntegrationPattern(Enum):
    RECTANGLE = 1
    TRAPEZOID = 2
    SIMPSON = 3
    MONTE_CARLO = 4
    LEGENDRE = 5
    HERMITE = 6
    LAGUERRE = 7
    CHEBYSHEV = 8
