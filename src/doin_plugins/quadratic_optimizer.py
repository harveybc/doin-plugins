"""Quadratic Optimizer — reference optimization plugin.

Optimizes parameters to minimize f(x) = sum((x_i - target_i)^2).
This is intentionally simple to test the full DON pipeline without
requiring ML frameworks.

The optimizer uses a simple hill-climbing approach with random perturbations.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from doin_core.plugins.base import OptimizationPlugin


class QuadraticOptimizer(OptimizationPlugin):
    """Optimizes parameters for a simple quadratic function.

    Configuration:
        n_params: Number of parameters to optimize (default: 10)
        target: Target values (default: random in [-5, 5])
        step_size: Perturbation size (default: 0.5)
        seed: Random seed (default: None)
    """

    def __init__(self) -> None:
        self._n_params = 10
        self._target: np.ndarray | None = None
        self._step_size = 0.5
        self._rng = random.Random()

    def configure(self, config: dict[str, Any]) -> None:
        self._n_params = config.get("n_params", 10)
        self._step_size = config.get("step_size", 0.5)

        seed = config.get("seed")
        if seed is not None:
            self._rng = random.Random(seed)
            np.random.seed(seed)

        target = config.get("target")
        if target is not None:
            self._target = np.array(target, dtype=np.float64)
        else:
            self._target = np.random.uniform(-5, 5, self._n_params)

    def optimize(
        self,
        current_best_params: dict[str, Any] | None,
        current_best_performance: float | None,
    ) -> tuple[dict[str, Any], float]:
        if self._target is None:
            self._target = np.random.uniform(-5, 5, self._n_params)

        # Start from current best or random
        if current_best_params is not None:
            x = np.array(current_best_params["x"], dtype=np.float64)
        else:
            x = np.random.uniform(-10, 10, self._n_params)

        # Hill climbing: perturb and keep if better
        perturbation = np.array(
            [self._rng.gauss(0, self._step_size) for _ in range(self._n_params)]
        )
        candidate = x + perturbation

        current_loss = float(np.sum((x - self._target) ** 2))
        candidate_loss = float(np.sum((candidate - self._target) ** 2))

        # Performance = negative loss (higher is better)
        if candidate_loss < current_loss or current_best_params is None:
            performance = -candidate_loss
            return {"x": candidate.tolist()}, performance
        else:
            # Return current with same performance — runner will filter as no improvement
            performance = -current_loss
            return {"x": x.tolist()}, performance

    def get_domain_metadata(self) -> dict[str, Any]:
        return {
            "performance_metric": "neg_mse",
            "higher_is_better": True,
        }
