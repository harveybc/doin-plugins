"""Quadratic Inferencer — reference inference plugin.

Evaluates parameters against the quadratic function f(x) = sum((x_i - target_i)^2).
Returns negative MSE as the performance metric (higher is better).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from doin_core.plugins.base import InferencePlugin


class QuadraticInferencer(InferencePlugin):
    """Evaluates parameters for the quadratic optimization domain.

    Configuration:
        target: Target values for the quadratic function.
    """

    def __init__(self) -> None:
        self._target: np.ndarray | None = None

    def configure(self, config: dict[str, Any]) -> None:
        target = config.get("target")
        if target is not None:
            self._target = np.array(target, dtype=np.float64)

    def evaluate(
        self,
        parameters: dict[str, Any],
        data: dict[str, Any] | None = None,
    ) -> float:
        x = np.array(parameters["x"], dtype=np.float64)

        # Use target from data if provided (synthetic data case),
        # otherwise use configured target
        if data is not None and "target" in data:
            target = np.array(data["target"], dtype=np.float64)
        elif self._target is not None:
            target = self._target
        else:
            raise ValueError("No target configured and no target in data")

        loss = float(np.sum((x - target) ** 2))
        return -loss  # Negative MSE — higher is better
