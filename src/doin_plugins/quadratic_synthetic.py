"""Quadratic Synthetic Data — reference synthetic data generation plugin.

Generates slightly noisy versions of the target to test evaluator
verification without using the exact training/validation data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from doin_core.plugins.base import SyntheticDataPlugin


class QuadraticSyntheticData(SyntheticDataPlugin):
    """Generates synthetic target data for the quadratic domain.

    Configuration:
        target: Base target values.
        noise_std: Standard deviation of noise added to target (default: 0.1).
    """

    def __init__(self) -> None:
        self._target: np.ndarray | None = None
        self._noise_std = 0.1

    def configure(self, config: dict[str, Any]) -> None:
        target = config.get("target")
        if target is not None:
            self._target = np.array(target, dtype=np.float64)
        self._n_params = config.get("n_params", 10)
        self._noise_std = config.get("noise_std", 0.1)

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        rng = np.random.default_rng(seed)

        # If no target configured (e.g. evaluator on a different node),
        # generate a deterministic target from the seed. This ensures
        # each evaluator gets unique but reproducible synthetic data.
        if self._target is None:
            base_target = rng.uniform(-5, 5, self._n_params)
        else:
            base_target = self._target

        noisy_target = base_target + rng.normal(0, self._noise_std, len(base_target))

        return {
            "target": noisy_target.tolist(),
            "synthetic": True,
            "noise_std": self._noise_std,
        }
