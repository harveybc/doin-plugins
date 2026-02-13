"""Synthetic data generator for the predictor domain.

Wraps harveybc/timeseries-gan (SDG/TSG) as a DON SyntheticDataPlugin.
Uses a trained SC-VAE-GAN to generate timeseries data that preserves
the statistical properties of the real training data but with different
actual values — making it impossible to memorize the verification data.

CRITICAL: generate() is DETERMINISTIC given the same seed.
All evaluators in a quorum use the same seed (derived from commitment hash)
→ same synthetic data → same verification results → quorum consensus works.

The synthetic data hash is included in each evaluator's vote and checked
by the quorum to ensure all evaluators used identical data.

Dependencies:
  - harveybc/timeseries-gan (pip install -e /path/to/timeseries-gan)
  - Pre-trained GAN models (generator, encoder, decoder .keras files)

If timeseries-gan is not installed, falls back to a simpler block-bootstrap
generator (less realistic but still deterministic and unpredictable).

Configuration:
  tsg_root:       Path to timeseries-gan repo (auto-detected)
  generator_model: Path to trained generator .keras model
  encoder_model:   Path to trained encoder .keras model
  decoder_model:   Path to trained decoder .keras model
  n_samples:       Number of samples to generate (default: match val set size)
  predictor_root:  Path to predictor repo (for fallback data loading)
  load_config:     Path to predictor config JSON
  method:          "gan" (default, uses TSG) or "bootstrap" (fallback)
  block_size:      Block size for bootstrap fallback (default: 50)
  noise_scale:     Noise scale for bootstrap fallback (default: 0.05)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from doin_core.plugins.base import SyntheticDataPlugin


class PredictorSyntheticData(SyntheticDataPlugin):
    """Generates synthetic timeseries data for predictor verification.

    Primary method: wraps timeseries-gan (SC-VAE-GAN) for high-quality
    synthetic data that matches the real data distribution.

    Fallback method: block bootstrap with noise (if TSG not installed).
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._method = "gan"
        self._tsg_root: Path | None = None
        self._predictor_root: Path | None = None
        self._predictor_config: dict[str, Any] = {}

        # GAN-specific state
        self._tsg_available = False
        self._generator_model: Any = None
        self._feeder_plugin: Any = None
        self._generator_plugin: Any = None
        self._tsg_config: dict[str, Any] = {}

        # Fallback state
        self._real_data: dict[str, Any] | None = None
        self._block_size = 50
        self._noise_scale = 0.05
        self._n_samples: int | None = None

    def configure(self, config: dict[str, Any]) -> None:
        self._config = config
        self._method = config.get("method", "gan")
        self._block_size = config.get("block_size", 50)
        self._noise_scale = config.get("noise_scale", 0.05)
        self._n_samples = config.get("n_samples")

        # Resolve timeseries-gan repo
        tsg_root = config.get("tsg_root")
        if tsg_root:
            self._tsg_root = Path(tsg_root).resolve()
        else:
            for candidate in [
                Path.home() / "timeseries-gan",
                Path.cwd() / "timeseries-gan",
                Path("/home/openclaw/timeseries-gan"),
            ]:
                if (candidate / "setup.py").exists():
                    self._tsg_root = candidate
                    break

        # Resolve predictor repo (for fallback data loading)
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

        # Try to load timeseries-gan
        if self._method == "gan" and self._tsg_root is not None:
            try:
                self._setup_tsg()
                self._tsg_available = True
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "timeseries-gan not available, falling back to bootstrap: %s", e
                )
                self._method = "bootstrap"
                self._tsg_available = False

        # For bootstrap fallback, load real data
        if self._method == "bootstrap":
            self._load_real_data()

    def _setup_tsg(self) -> None:
        """Initialize timeseries-gan generator."""
        root_str = str(self._tsg_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        # Load TSG config
        tsg_config_path = self._config.get("tsg_config")
        if tsg_config_path:
            with open(tsg_config_path) as f:
                self._tsg_config = json.load(f)
        else:
            # Use TSG defaults
            from app.config import DEFAULT_VALUES as TSG_DEFAULTS
            self._tsg_config = TSG_DEFAULTS.copy()

        # Load trained models
        generator_model_path = self._config.get("generator_model")
        if generator_model_path:
            self._tsg_config["generator_model"] = generator_model_path

        # Load feeder and generator plugins from TSG
        from app.plugin_loader import load_plugin as tsg_load_plugin

        feeder_name = self._tsg_config.get("feeder_plugin", "default_feeder")
        feeder_cls, _ = tsg_load_plugin("feeder.plugins", feeder_name)
        self._feeder_plugin = feeder_cls()

        gen_name = self._tsg_config.get("generator_plugin", "default_generator")
        gen_cls, _ = tsg_load_plugin("generator.plugins", gen_name)
        self._generator_plugin = gen_cls()

        # Configure plugins
        if hasattr(self._feeder_plugin, "set_params"):
            self._feeder_plugin.set_params(**self._tsg_config)
        if hasattr(self._generator_plugin, "set_params"):
            self._generator_plugin.set_params(**self._tsg_config)

    def _load_real_data(self) -> None:
        """Load real data for bootstrap fallback."""
        if self._predictor_root is None:
            raise FileNotFoundError(
                "Cannot find predictor repo for bootstrap fallback. "
                "Set 'predictor_root' in config."
            )

        root_str = str(self._predictor_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        from app.config import DEFAULT_VALUES
        pred_config = DEFAULT_VALUES.copy()

        config_file = self._config.get("load_config")
        if config_file:
            config_path = Path(config_file)
            if not config_path.is_absolute():
                config_path = self._predictor_root / config_path
            if config_path.exists():
                with open(config_path) as f:
                    pred_config.update(json.load(f))

        from app.plugin_loader import load_plugin

        pre_name = pred_config.get("preprocessor_plugin", "default_preprocessor")
        pre_cls, _ = load_plugin("preprocessor.plugins", pre_name)
        preprocessor = pre_cls()
        preprocessor.set_params(**pred_config)

        tgt_name = pred_config.get("target_plugin", "default_target")
        tgt_cls, _ = load_plugin("target.plugins", tgt_name)
        target = tgt_cls()
        target.set_params(**pred_config)

        datasets = preprocessor.run_preprocessing(target, pred_config)
        if isinstance(datasets, tuple):
            datasets = datasets[0]

        self._real_data = {
            "x_train": np.asarray(datasets["x_train"]),
            "y_train": np.asarray(datasets["y_train"]),
            "x_val": np.asarray(datasets["x_val"]),
            "y_val": np.asarray(datasets["y_val"]),
            "baseline_val": (
                np.asarray(datasets["baseline_val"])
                if datasets.get("baseline_val") is not None else None
            ),
        }

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        """Generate synthetic data deterministically.

        DETERMINISTIC: same seed → identical output across all evaluators.
        This is verified by the quorum via synthetic data hash consensus.

        Args:
            seed: Random seed. When used in quorum verification, this is
                  derived from commitment_hash + domain_id so all evaluators
                  get the same seed automatically.

        Returns:
            Dict with x_train, y_train, x_val, y_val, baseline_val keys.
        """
        if self._tsg_available and self._method == "gan":
            return self._generate_gan(seed)
        else:
            return self._generate_bootstrap(seed)

    def _generate_gan(self, seed: int | None = None) -> dict[str, Any]:
        """Generate using timeseries-gan (SC-VAE-GAN).

        Sets all random seeds (numpy, tensorflow) for determinism.
        """
        # Set all seeds for deterministic generation
        if seed is not None:
            np.random.seed(seed)
            try:
                import tensorflow as tf
                tf.random.set_seed(seed)
            except ImportError:
                pass

        rng = np.random.default_rng(seed)

        # Determine number of samples
        n_samples = self._n_samples
        if n_samples is None and self._real_data is not None:
            n_samples = len(self._real_data["x_val"])
        if n_samples is None:
            n_samples = 500  # Default

        # Use TSG's SyntheticDataGenerator
        from app.data_generation.synthetic_generator import SyntheticDataGenerator

        generator = SyntheticDataGenerator(
            config=self._tsg_config,
            feeder_plugin=self._feeder_plugin,
            generator_plugin=self._generator_plugin,
        )

        # Generate synthetic data
        synthetic_df = generator.generate(n_samples=n_samples)

        # Convert to predictor-compatible format
        # TSG outputs a DataFrame; we need numpy arrays matching predictor's format
        values = synthetic_df.select_dtypes(include=[np.number]).values.astype(np.float32)

        # Split into train/val (70/30)
        split_idx = int(len(values) * 0.7)

        # For the predictor domain, we need windowed data
        # Use a simple sliding window to create x/y pairs
        window_size = self._config.get("window_size", 30)
        result = self._create_windowed_data(values, window_size, split_idx, rng)
        result["synthetic"] = True
        result["method"] = "gan"
        result["seed"] = seed

        return result

    def _generate_bootstrap(self, seed: int | None = None) -> dict[str, Any]:
        """Block bootstrap: resample blocks of real data with noise.

        Deterministic given the same seed.
        """
        if self._real_data is None:
            raise ValueError("Real data not loaded — call configure() first")

        rng = np.random.default_rng(seed)
        result: dict[str, Any] = {"synthetic": True, "method": "bootstrap", "seed": seed}

        for key in ["x_train", "y_train", "x_val", "y_val"]:
            real = self._real_data[key]
            n = len(real)
            bs = min(self._block_size, n)

            if bs <= 0 or n <= 0:
                result[key] = real.copy()
                continue

            # Block bootstrap with deterministic rng
            indices = []
            while len(indices) < n:
                start = int(rng.integers(0, max(1, n - bs)))
                indices.extend(range(start, min(start + bs, n)))
            indices = indices[:n]

            synthetic = real[indices].copy()

            # Add small noise for uniqueness
            if synthetic.dtype in (np.float32, np.float64):
                noise_std = float(np.std(synthetic)) * self._noise_scale
                synthetic = synthetic + rng.normal(0, noise_std, synthetic.shape).astype(synthetic.dtype)

            result[key] = synthetic

        # Baseline
        if self._real_data.get("baseline_val") is not None:
            baseline = self._real_data["baseline_val"]
            n_b = len(baseline)
            indices = [int(x) for x in rng.integers(0, n_b, size=n_b)]
            result["baseline_val"] = baseline[indices].copy()
        else:
            result["baseline_val"] = None

        return result

    def _create_windowed_data(
        self,
        values: np.ndarray,
        window_size: int,
        split_idx: int,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Create windowed x/y pairs from raw synthetic data."""
        n_features = values.shape[1] if len(values.shape) > 1 else 1
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)

        # Create windows
        x_all, y_all = [], []
        for i in range(len(values) - window_size):
            x_all.append(values[i:i + window_size])
            y_all.append(values[i + window_size, :1])  # Predict first feature

        x_all = np.array(x_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.float32)

        # Split
        n_train = min(split_idx, len(x_all))
        return {
            "x_train": x_all[:n_train],
            "y_train": y_all[:n_train],
            "x_val": x_all[n_train:],
            "y_val": y_all[n_train:],
            "baseline_val": y_all[n_train:],  # Naive baseline
        }
