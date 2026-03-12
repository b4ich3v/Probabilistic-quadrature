from enum import Enum


class MonteCarloIntegrationStrategy(Enum):
    STANDARD = 0
    WEIGHTED = 1
    RECURSIVE = 2