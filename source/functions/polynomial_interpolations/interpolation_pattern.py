from enum import Enum


class InterpolationPattern(Enum):
    LAGRANGE = 0
    NEWTON = 1
    HERMITE = 2