"""Synthetic data generator for the predictor domain.

Uses block bootstrap to generate typical_price timeseries data that
preserves statistical properties of real data but with different values —
making it impossible to memorize the verification data.

CRITICAL: generate() is DETERMINISTIC given the same seed.
All evaluators in a quorum use the same seed (derived from commitment hash)
→ same synthetic data → same verification results → quorum consensus works.

The synthetic data hash is included in each evaluator's vote and checked
by the quorum to ensure all evaluators used identical data.

Method: Block Bootstrap (proven best in augmentation sweep):
  - Resamples contiguous blocks from real training data (d1-d3)
  - Preserves local temporal structure within blocks
  - Different seed → different block arrangement → different data
  - No training required — fast, deterministic, reproducible

Sweep results (2026-02-16):
  - bb_n1000 (bs=30): val +4.39%, test +0.61%  ← best block bootstrap
  - tg_n500:          val +3.25%, test +1.09%  ← best TimeGAN
  - All other configs hurt test performance

Configuration:
  n_samples:       Number of 4h typical_price rows (default: 1560 = 1 year)
  block_size:      Contiguous block length for bootstrap (default: 30)
  predictor_root:  Path to predictor repo (for loading real data)
  data_files:      List of CSV paths for source data (overrides auto-detect)
  quiet:           Suppress verbose output (default: True)
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from doin_core.plugins.base import SyntheticDataPlugin

log = logging.getLogger(__name__)

# 1 year of 4h forex data (no weekends): 260 trading days × 6 candles/day
SAMPLES_PER_YEAR = 1560
DEFAULT_BLOCK_SIZE = 30  # Proven optimal in sweep


class PredictorSyntheticData(SyntheticDataPlugin):
    """Generates synthetic timeseries data for predictor verification.

    Method: Block bootstrap from real training data (d1-d3).
    Fast, deterministic, no model training required.
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._predictor_root: Path | None = None
        self._real_prices: np.ndarray | None = None
        self._n_samples = SAMPLES_PER_YEAR
        self._block_size = DEFAULT_BLOCK_SIZE

    def configure(self, config: dict[str, Any]) -> None:
        self._config = config
        self._n_samples = config.get("n_samples", SAMPLES_PER_YEAR)
        self._block_size = config.get("block_size", DEFAULT_BLOCK_SIZE)

        # Resolve predictor repo
        root = config.get("predictor_root")
        if root:
            self._predictor_root = Path(root).resolve()
        else:
            for candidate in [
                Path.home() / "predictor",
                Path.cwd() / "predictor",
                Path("/home/openclaw/predictor"),
            ]:
                if (candidate / "setup.py").exists():
                    self._predictor_root = candidate
                    break

        if self._predictor_root is None:
            raise FileNotFoundError(
                "Cannot find predictor repo. Set 'predictor_root' in config."
            )

        self._load_real_data(config.get("data_files"))

    def _load_real_data(self, data_files: list[str] | None = None) -> None:
        """Load real training data (d1-d3) for block bootstrap source.

        Only d1-d3 are used — these are the generator training sets.
        d4/d5/d6 are predictor train/val/test and must never be seen by
        the synthetic generator.
        """
        if data_files:
            paths = [Path(f) for f in data_files]
        else:
            data_dir = self._predictor_root / "examples" / "data_downsampled" / "phase_1"
            paths = [data_dir / f"base_d{i}.csv" for i in range(1, 4)]

        all_prices = []
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Source data not found: {p}")
            df = pd.read_csv(p)
            col = "typical_price" if "typical_price" in df.columns else df.columns[-1]
            all_prices.append(df[col].values)

        self._real_prices = np.concatenate(all_prices)
        log.info(
            "Loaded %d real price samples from %d files for bootstrap",
            len(self._real_prices), len(paths),
        )

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        """Generate synthetic data deterministically via block bootstrap.

        DETERMINISTIC: same seed → identical output across all evaluators.
        Verified by quorum via synthetic data hash consensus.

        Args:
            seed: Random seed. In quorum verification, derived from
                  hash(commitment + domain + evaluator_id + chain_tip_hash).

        Returns:
            Dict with:
              - synthetic_df: DataFrame with DATE_TIME, typical_price columns
              - synthetic_csv: CSV string for hashing
              - data_hash: SHA256 hash of synthetic data (for quorum consensus)
              - n_samples: number of generated rows
              - method: "block_bootstrap"
              - seed: the seed used
              - block_size: block size used
        """
        if self._real_prices is None:
            raise RuntimeError("Real data not loaded — call configure() first")

        seed = seed or 42
        rng = np.random.default_rng(seed)

        n_real = len(self._real_prices)
        bs = min(self._block_size, n_real)

        # Block bootstrap: sample random starting positions, take contiguous blocks
        indices = []
        while len(indices) < self._n_samples:
            start = int(rng.integers(0, max(1, n_real - bs)))
            indices.extend(range(start, start + bs))
        indices = indices[:self._n_samples]

        synthetic_prices = self._real_prices[indices].copy()

        # Build DataFrame with synthetic timestamps (4h intervals, skip weekends)
        dates = self._generate_trading_dates(self._n_samples, rng)
        df = pd.DataFrame({
            "DATE_TIME": dates,
            "typical_price": synthetic_prices,
        })

        # Create CSV string for hashing
        csv_str = df.to_csv(index=False)
        data_hash = hashlib.sha256(csv_str.encode()).hexdigest()

        log.info(
            "Generated %d synthetic samples (block_size=%d, seed=%d, hash=%s…)",
            len(df), bs, seed, data_hash[:12],
        )

        return {
            "synthetic_df": df,
            "synthetic_csv": csv_str,
            "data_hash": data_hash,
            "n_samples": len(df),
            "method": "block_bootstrap",
            "seed": seed,
            "block_size": bs,
        }

    def get_data_hash(self, seed: int) -> str:
        """Quick hash computation for quorum pre-verification.

        Evaluators can exchange hashes before full evaluation to confirm
        they're all using identical synthetic data.
        """
        result = self.generate(seed=seed)
        return result["data_hash"]

    @staticmethod
    def _generate_trading_dates(n: int, rng: np.random.Generator) -> list:
        """Generate n trading timestamps at 4h intervals, skipping weekends."""
        from datetime import datetime, timedelta

        dates = []
        # Start from a known Monday
        dt = datetime(2020, 1, 6, 0, 0, 0)  # Monday
        while len(dates) < n:
            if dt.weekday() < 5:  # Mon-Fri
                dates.append(dt)
            dt += timedelta(hours=4)
            # Skip to Monday if we hit Saturday
            if dt.weekday() == 5:
                dt += timedelta(days=2)
        return dates[:n]
