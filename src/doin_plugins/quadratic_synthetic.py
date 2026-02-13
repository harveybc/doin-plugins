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
        self._noise_std = config.get("noise_std", 0.1)

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        if self._target is None:
            raise ValueError("No target configured")

        rng = np.random.default_rng(seed)
        noisy_target = self._target + rng.normal(0, self._noise_std, len(self._target))

        return {
            "target": noisy_target.tolist(),
            "synthetic": True,
            "noise_std": self._noise_std,
        }
