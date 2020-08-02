from typing import Optional

import torch.optim as optim


class WarmupOptimizer:
    """Optim wrapper that implements rate."""

    def __init__(self, base_optimizer: optim.Optimizer, d_model: int, scale_factor: float, warmup_steps: int):
        self.base_optimizer = base_optimizer
        self._step = 0
        self.warmup_steps = warmup_steps
        self.scale_factor = scale_factor
        self.d_model = d_model
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.base_optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.base_optimizer.step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def rate(self, step: Optional[int] = None) -> float:
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.scale_factor * (self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
