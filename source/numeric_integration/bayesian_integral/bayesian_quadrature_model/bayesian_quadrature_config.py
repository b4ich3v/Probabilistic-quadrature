from dataclasses import dataclass


@dataclass
class BQConfig:
    noise: float = 0.0
    jitter: float = 1e-8
    mc_samples_mean: int = 2048
    mc_samples_var: int = 4096

    def validate(self) -> None:
        if self.noise < 0:
            raise ValueError("noise must be non-negative")
        if self.jitter < 0:
            raise ValueError("jitter must be non-negative")
        if self.mc_samples_mean <= 0 or self.mc_samples_var <= 0:
            raise ValueError("mc_samples_* must be positive")
