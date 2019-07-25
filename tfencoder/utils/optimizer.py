from typing import Optional

import torch.optim as optim


class TFOptimizer:
    """Optim wrapper that implements rate."""

    def __init__(
        self, d_model: int, factor: float, warmup: int, optimizer: optim.Optimizer
    ) -> None:
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step: Optional[int] = None) -> float:
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (
            self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )
