from enum import Enum


class NumericIntegrationPattern(Enum):
    RECTANGLE = 1
    TRAPEZOID = 2
    SIMPSON = 3
    MONTE_CARLO = 4