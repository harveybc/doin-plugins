"""Synthetic data generator for the predictor domain.

Loads a PRE-TRAINED HMM hybrid generator model and uses it to produce
deterministic synthetic typical_price timeseries for DOIN evaluator
verification.

CRITICAL ARCHITECTURE:
  - The generator is trained ONCE offline on d1-d4 with optimized params
  - The trained model is saved as a .joblib file
  - Evaluators ONLY LOAD the model and call generate() — they NEVER retrain
  - This mirrors how predictors work: train once → export model → load for inference

CRITICAL: generate() is DETERMINISTIC given the same seed.
All evaluators in a quorum use the same seed (derived from commitment hash)
→ same synthetic data → same verification results → quorum consensus works.

The synthetic data hash is included in each evaluator's vote and checked
by the quorum to ensure all evaluators used identical data.

Pre-trained model: hmm_hybrid_d1d4_optimized.joblib
  - Trained on d1-d4 raw prices (18,462 samples)
  - Optimized config (composite score 0.1984, 32% better than default):
    n_regimes=5, block_size=30, smooth_weight=0.5,
    min_block_length=2, covariance_type=diag, vol_windows=(6,16,48)
  - Tolerance eval: 9.2% mean test deviation

Configuration:
  n_samples:     Number of 4h typical_price rows (default: 1560 = 1 year)
  model_file:    Path to pre-trained .joblib model (required)
  sdg_root:      Path to synthetic-datagen repo (for generator code)
  initial_price: Starting price for generation (default: 1.2, ~EUR/USD)
  quiet:         Suppress verbose output (default: True)
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

# Default initial price (~EUR/USD range)
DEFAULT_INITIAL_PRICE = 1.2


class PredictorSyntheticData(SyntheticDataPlugin):
    """Loads a pre-trained HMM hybrid model and generates synthetic data.

    The model file is produced offline by training on d1-d4 with optimized
    parameters. Evaluators load it and call generate(seed) — fast, deterministic,
    no training involved.
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._hybrid_model: dict | None = None
        self._hybrid_generate = None
        self._n_samples = SAMPLES_PER_YEAR
        self._initial_price = DEFAULT_INITIAL_PRICE

    def configure(self, config: dict[str, Any]) -> None:
        self._config = config
        self._n_samples = config.get("n_samples", SAMPLES_PER_YEAR)
        self._initial_price = config.get("initial_price", DEFAULT_INITIAL_PRICE)

        # Resolve synthetic-datagen repo (needed for generator code)
        sdg_root = config.get("sdg_root")
        if sdg_root:
            sdg_path = Path(sdg_root).resolve()
        else:
            for candidate in [
                Path.home() / ".openclaw/workspace/synthetic-datagen",
                Path.home() / "synthetic-datagen",
                Path.cwd() / "synthetic-datagen",
            ]:
                if (candidate / "sdg_plugins").exists():
                    sdg_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    "Cannot find synthetic-datagen repo. Set 'sdg_root' in config."
                )

        if str(sdg_path) not in sys.path:
            sys.path.insert(0, str(sdg_path))

        # Import generator function
        from sdg_plugins.generator.regime_bootstrap_hybrid import generate as hybrid_generate
        self._hybrid_generate = hybrid_generate

        # Load pre-trained model
        model_file = config.get("model_file")
        if not model_file:
            # Default location
            model_file = sdg_path / "examples" / "models" / "hmm_hybrid_d1d4_optimized.joblib"
        model_path = Path(model_file).resolve()

        if not model_path.exists():
            raise FileNotFoundError(
                f"Pre-trained generator model not found: {model_path}\n"
                "Train it first with: synthetic-datagen/examples/scripts/train_generator.py"
            )

        import joblib
        self._hybrid_model = joblib.load(model_path)
        log.info(
            "Loaded pre-trained HMM hybrid model from %s "
            "(K=%d, block_size=%d, covariance=%s)",
            model_path,
            self._hybrid_model.get("n_regimes", "?"),
            self._hybrid_model.get("block_size", "?"),
            self._hybrid_model.get("covariance_type", "?"),
        )

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        """Generate synthetic data deterministically from pre-trained model.

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
              - method: "hmm_hybrid"
              - seed: the seed used
        """
        if self._hybrid_model is None:
            raise RuntimeError("Model not loaded — call configure() first")

        seed = seed or 42

        synthetic_prices = self._hybrid_generate(
            self._hybrid_model, self._n_samples,
            seed=seed, initial_price=self._initial_price,
        )

        # Build DataFrame with synthetic timestamps (4h intervals, skip weekends)
        dates = self._generate_trading_dates(self._n_samples)
        df = pd.DataFrame({
            "DATE_TIME": dates,
            "typical_price": synthetic_prices[:self._n_samples],
        })

        # Create CSV string for hashing
        csv_str = df.to_csv(index=False)
        data_hash = hashlib.sha256(csv_str.encode()).hexdigest()

        log.info(
            "Generated %d synthetic samples (seed=%d, hash=%s…)",
            len(df), seed, data_hash[:12],
        )

        return {
            "synthetic_df": df,
            "synthetic_csv": csv_str,
            "data_hash": data_hash,
            "n_samples": len(df),
            "method": "hmm_hybrid",
            "seed": seed,
        }

    def get_data_hash(self, seed: int) -> str:
        """Quick hash computation for quorum pre-verification."""
        result = self.generate(seed=seed)
        return result["data_hash"]

    @staticmethod
    def _generate_trading_dates(n: int) -> list:
        """Generate n trading timestamps at 4h intervals, skipping weekends."""
        from datetime import datetime, timedelta

        dates = []
        dt = datetime(2020, 1, 6, 0, 0, 0)  # Known Monday
        while len(dates) < n:
            if dt.weekday() < 5:  # Mon-Fri
                dates.append(dt)
            dt += timedelta(hours=4)
            if dt.weekday() == 5:  # Skip weekends
                dt += timedelta(days=2)
        return dates[:n]
