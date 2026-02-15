"""Synthetic data generator for the predictor domain.

Wraps harveybc/synthetic-datagen as a DOIN SyntheticDataPlugin.
Uses a trained VAE-GAN to generate typical_price timeseries data that
preserves statistical properties of real data but with different values —
making it impossible to memorize the verification data.

CRITICAL: generate() is DETERMINISTIC given the same seed.
All evaluators in a quorum use the same seed (derived from commitment hash)
→ same synthetic data → same verification results → quorum consensus works.

The synthetic data hash is included in each evaluator's vote and checked
by the quorum to ensure all evaluators used identical data.

Dependencies:
  - harveybc/synthetic-datagen (pip install -e /path/to/synthetic-datagen)
  - Pre-trained VAE-GAN model (.keras + .keras.parts/)

Configuration:
  generator_model:  Path to trained .keras model (required)
  n_samples:        Number of 4h typical_price rows to generate (default: 2190 = ~1 year)
  sdg_root:         Path to synthetic-datagen repo (auto-detected)
  predictor_root:   Path to predictor repo (for fallback data loading)
  load_config:      Path to predictor config JSON
  method:           "sdg" (default) or "bootstrap" (fallback)
  block_size:       Block size for bootstrap fallback (default: 50)
  noise_scale:      Noise scale for bootstrap fallback (default: 0.05)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from doin_core.plugins.base import SyntheticDataPlugin

log = logging.getLogger(__name__)


class PredictorSyntheticData(SyntheticDataPlugin):
    """Generates synthetic timeseries data for predictor verification.

    Primary method: wraps synthetic-datagen's TypicalPriceGenerator for
    high-quality VAE-GAN synthetic data.

    Fallback method: block bootstrap with noise (if synthetic-datagen not available).
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._method = "sdg"
        self._sdg_root: Path | None = None
        self._predictor_root: Path | None = None

        # SDG generator (reusable — load model once, generate many times)
        self._sdg_available = False
        self._generator = None

        # Fallback state
        self._real_data: dict[str, Any] | None = None
        self._block_size = 50
        self._noise_scale = 0.05
        self._n_samples = 2190  # ~1 year of 4h data

    def configure(self, config: dict[str, Any]) -> None:
        self._config = config
        self._method = config.get("method", "sdg")
        self._block_size = config.get("block_size", 50)
        self._noise_scale = config.get("noise_scale", 0.05)
        self._n_samples = config.get("n_samples", 2190)

        # Resolve synthetic-datagen repo
        sdg_root = config.get("sdg_root")
        if sdg_root:
            self._sdg_root = Path(sdg_root).resolve()
        else:
            for candidate in [
                Path.home() / "synthetic-datagen",
                Path.cwd() / "synthetic-datagen",
                Path("/home/openclaw/.openclaw/workspace/synthetic-datagen"),
            ]:
                if (candidate / "pyproject.toml").exists():
                    self._sdg_root = candidate
                    break

        # Resolve predictor repo (for fallback)
        pred_root = config.get("predictor_root")
        if pred_root:
            self._predictor_root = Path(pred_root).resolve()
        else:
            for candidate in [
                Path.home() / "predictor",
                Path.cwd() / "predictor",
                Path("/home/openclaw/predictor"),
            ]:
                if (candidate / "setup.py").exists():
                    self._predictor_root = candidate
                    break

        # Try to load synthetic-datagen generator
        if self._method == "sdg":
            try:
                self._setup_sdg()
                self._sdg_available = True
                log.info("synthetic-datagen generator loaded successfully")
            except Exception as e:
                log.warning(
                    "synthetic-datagen not available, falling back to bootstrap: %s", e
                )
                self._method = "bootstrap"
                self._sdg_available = False

        # For bootstrap fallback, load real data
        if self._method == "bootstrap":
            self._load_real_data()

    def _setup_sdg(self) -> None:
        """Initialize synthetic-datagen generator and load model."""
        # Ensure synthetic-datagen is importable
        if self._sdg_root:
            root_str = str(self._sdg_root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)

        from sdg_plugins.generator.typical_price_generator import TypicalPriceGenerator

        model_path = self._config.get("generator_model")
        if not model_path:
            # Try default location
            if self._sdg_root:
                default = self._sdg_root / "examples" / "models" / "vae_gan_4h.keras"
                if default.exists():
                    model_path = str(default)

        if not model_path:
            raise FileNotFoundError(
                "No generator_model path configured and no default model found. "
                "Train one with: sdg --mode train --train_data d1.csv d2.csv d3.csv"
            )

        # Create generator with minimal config
        gen_config = {
            "load_model": model_path,
            "n_samples": self._n_samples,
            "seed": 42,  # will be overridden per-call
            "use_returns": self._config.get("use_returns", True),
            "interval_hours": 4,
            "start_datetime": "2020-01-01 00:00:00",
        }
        self._generator = TypicalPriceGenerator(gen_config)
        self._generator.load_model(model_path)
        log.info(f"Loaded VAE-GAN model from {model_path}")

    def _load_real_data(self) -> None:
        """Load real data for bootstrap fallback."""
        if self._predictor_root is None:
            raise FileNotFoundError(
                "Cannot find predictor repo for bootstrap fallback. "
                "Set 'predictor_root' in config."
            )

        # Load CSV files directly
        data_dir = self._predictor_root / "examples" / "data_downsampled" / "phase_1"
        train_path = data_dir / "base_d1.csv"
        val_path = data_dir / "base_d2.csv"
        test_path = data_dir / "base_d3.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        self._real_data = {
            "train": pd.read_csv(train_path, parse_dates=["DATE_TIME"]),
            "val": pd.read_csv(val_path, parse_dates=["DATE_TIME"]),
            "test": pd.read_csv(test_path, parse_dates=["DATE_TIME"]),
        }

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        """Generate synthetic data deterministically.

        DETERMINISTIC: same seed → identical output across all evaluators.
        This is verified by the quorum via synthetic data hash consensus.

        Args:
            seed: Random seed. In quorum verification, derived from
                  hash(commitment + domain + evaluator_id + chain_tip_hash).

        Returns:
            Dict with:
              - synthetic_df: DataFrame with DATE_TIME, typical_price columns
              - synthetic_csv: CSV string for hashing
              - data_hash: SHA256 hash of synthetic data (for quorum consensus)
              - n_samples: number of generated rows
              - method: "sdg" or "bootstrap"
              - seed: the seed used
        """
        if self._sdg_available and self._method == "sdg":
            return self._generate_sdg(seed)
        else:
            return self._generate_bootstrap(seed)

    def _generate_sdg(self, seed: int | None = None) -> dict[str, Any]:
        """Generate using synthetic-datagen VAE-GAN.

        Deterministic: same seed + same model = identical output.
        """
        if self._generator is None:
            raise RuntimeError("Generator not initialized — call configure() first")

        seed = seed or 42
        df = self._generator.generate(seed=seed, n_samples=self._n_samples, output_file="")

        # Create CSV string for hashing (ensures all evaluators can verify identical data)
        csv_str = df.to_csv(index=False)
        data_hash = hashlib.sha256(csv_str.encode()).hexdigest()

        return {
            "synthetic_df": df,
            "synthetic_csv": csv_str,
            "data_hash": data_hash,
            "n_samples": len(df),
            "method": "sdg",
            "seed": seed,
        }

    def _generate_bootstrap(self, seed: int | None = None) -> dict[str, Any]:
        """Block bootstrap: resample blocks of real data with noise.

        Deterministic given the same seed.
        """
        if self._real_data is None:
            raise ValueError("Real data not loaded — call configure() first")

        rng = np.random.default_rng(seed)

        # Bootstrap from training data
        train_prices = self._real_data["train"]["typical_price"].values
        n = len(train_prices)
        bs = min(self._block_size, n)

        # Block bootstrap
        indices = []
        while len(indices) < self._n_samples:
            start = int(rng.integers(0, max(1, n - bs)))
            indices.extend(range(start, min(start + bs, n)))
        indices = indices[:self._n_samples]

        synthetic_prices = train_prices[indices].copy()

        # Add noise
        noise_std = float(np.std(synthetic_prices)) * self._noise_scale
        synthetic_prices += rng.normal(0, noise_std, len(synthetic_prices))

        # Build DataFrame
        from datetime import datetime, timedelta
        start_dt = datetime(2020, 1, 1)
        dates = [start_dt + timedelta(hours=4 * i) for i in range(len(synthetic_prices))]

        df = pd.DataFrame({
            "DATE_TIME": dates,
            "typical_price": synthetic_prices,
        })

        csv_str = df.to_csv(index=False)
        data_hash = hashlib.sha256(csv_str.encode()).hexdigest()

        return {
            "synthetic_df": df,
            "synthetic_csv": csv_str,
            "data_hash": data_hash,
            "n_samples": len(df),
            "method": "bootstrap",
            "seed": seed,
        }

    def get_data_hash(self, seed: int) -> str:
        """Quick hash computation for quorum pre-verification.

        Evaluators can exchange hashes before full evaluation to confirm
        they're all using identical synthetic data.
        """
        result = self.generate(seed=seed)
        return result["data_hash"]
