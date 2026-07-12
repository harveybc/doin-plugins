"""Deterministic multi-asset scenario fixture for DOIN verification.

This is deliberately a transparent fixture, not a learned market generator.
The production scenario ladder can later replace the backend while preserving
the same ``SyntheticDataPlugin`` boundary and deterministic hash requirement.
"""

from __future__ import annotations

import hashlib
import io
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from doin_core.plugins.base import SyntheticDataPlugin


class TradingScenarioSyntheticData(SyntheticDataPlugin):
    """Generate deterministic aligned OHLCV rows for one or more assets."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}

    def configure(self, config: dict[str, Any]) -> None:
        self._config = dict(config)
        backend = str(config.get("scenario_backend", "fixture_v1"))
        if backend != "fixture_v1":
            raise ValueError(
                f"unsupported scenario_backend={backend!r}; only fixture_v1 "
                "is available until a versioned heuristic/gym-fx backend is promoted"
            )

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        seed = 0 if seed is None else int(seed)
        rng = np.random.default_rng(seed)
        assets = self._config.get("assets", [self._config.get("asset", "SOLUSDT")])
        assets = [str(asset) for asset in assets]
        rows = int(self._config.get("scenario_rows", 256))
        start = pd.Timestamp(self._config.get("scenario_start", "2020-01-06T00:00:00Z"))
        timeframe_hours = int(self._config.get("timeframe_hours", 4))
        timestamps = [start + timedelta(hours=timeframe_hours * i) for i in range(rows)]

        frames = []
        for index, asset in enumerate(assets):
            base = float(self._config.get("initial_prices", {}).get(asset, 100.0 + index * 25.0))
            returns = rng.normal(0.0, float(self._config.get("return_std", 0.01)), rows)
            close = base * np.exp(np.cumsum(returns))
            spread = np.abs(rng.normal(0.0, 0.002, rows))
            frame = pd.DataFrame({
                "DATE_TIME": timestamps,
                "asset": asset,
                "OPEN": close * (1.0 - spread),
                "HIGH": close * (1.0 + spread),
                "LOW": close * (1.0 - spread),
                "CLOSE": close,
                "VOLUME": rng.lognormal(10.0, 0.25, rows),
            })
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        csv_buffer = io.StringIO()
        frame.to_csv(csv_buffer, index=False, float_format="%.17g", lineterminator="\n")
        csv_text = csv_buffer.getvalue()
        return {
            "synthetic_df": frame,
            "synthetic_csv": csv_text,
            "data_hash": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
            "scenario_backend": "fixture_v1",
            "scenario_seed": seed,
            "assets": assets,
            "n_rows_per_asset": rows,
        }
