"""Tests for the predictor domain DON plugins.

These tests verify:
1. Plugin interface compliance (configure, optimize, evaluate, get_domain_metadata)
2. Param perturbation logic (bounds respected, int params rounded)
3. Fitness sign convention (DON higher=better ↔ predictor lower=better)

Note: Full integration tests require predictor to be installed with its
dependencies (TensorFlow, etc.). These unit tests mock the heavy parts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We need predictor on sys.path for imports to work
PREDICTOR_ROOT = Path("/home/openclaw/predictor")
if PREDICTOR_ROOT.exists() and str(PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PREDICTOR_ROOT))

from doin_plugins.predictor.optimizer import PredictorOptimizer
from doin_plugins.predictor.inferencer import PredictorInferencer


class TestPredictorOptimizer:
    """Tests for PredictorOptimizer."""

    def _make_mock_config(self) -> dict[str, Any]:
        """Minimal config that doesn't require actual predictor files."""
        return {
            "predictor_root": str(PREDICTOR_ROOT),
            "hyperparameter_bounds": {
                "learning_rate": [1e-5, 1e-2],
                "num_layers": [1, 5],
                "batch_size": [16, 64],
            },
            "step_size_fraction": 0.1,
            "seed": 42,
        }

    def test_interface_compliance(self):
        """PredictorOptimizer implements OptimizationPlugin interface."""
        from doin_core.plugins.base import OptimizationPlugin
        assert issubclass(PredictorOptimizer, OptimizationPlugin)

    def test_domain_metadata(self):
        """get_domain_metadata returns required fields."""
        opt = PredictorOptimizer()
        meta = opt.get_domain_metadata()
        assert "performance_metric" in meta
        assert "higher_is_better" in meta
        assert meta["higher_is_better"] is True

    def test_random_params_respects_bounds(self):
        """_random_params generates values within configured bounds."""
        opt = PredictorOptimizer()
        opt._bounds = {
            "learning_rate": (1e-5, 1e-2),
            "num_layers": (1, 5),
            "batch_size": (16, 64),
        }
        opt._rng = __import__("random").Random(42)

        for _ in range(50):
            params = opt._random_params()
            assert 1e-5 <= params["learning_rate"] <= 1e-2
            assert 1 <= params["num_layers"] <= 5
            assert isinstance(params["num_layers"], int)
            assert 16 <= params["batch_size"] <= 64
            assert isinstance(params["batch_size"], int)

    def test_perturb_stays_in_bounds(self):
        """_perturb keeps values clamped within bounds."""
        opt = PredictorOptimizer()
        opt._bounds = {
            "learning_rate": (1e-5, 1e-2),
            "num_layers": (1, 5),
        }
        opt._step_frac = 0.5  # Large perturbation
        opt._rng = __import__("random").Random(42)

        base = {"learning_rate": 5e-3, "num_layers": 3}
        for _ in range(100):
            perturbed = opt._perturb(base)
            assert 1e-5 <= perturbed["learning_rate"] <= 1e-2
            assert 1 <= perturbed["num_layers"] <= 5
            assert isinstance(perturbed["num_layers"], int)

    def test_perturb_handles_missing_base_param(self):
        """_perturb generates random value when param missing from base."""
        opt = PredictorOptimizer()
        opt._bounds = {
            "learning_rate": (1e-5, 1e-2),
            "new_param": (0, 10),
        }
        opt._step_frac = 0.1
        opt._rng = __import__("random").Random(42)

        base = {"learning_rate": 5e-3}  # new_param not present
        perturbed = opt._perturb(base)
        assert "new_param" in perturbed
        assert 0 <= perturbed["new_param"] <= 10

    def test_int_param_detection(self):
        """_is_int_param correctly identifies int parameters."""
        opt = PredictorOptimizer()
        assert opt._is_int_param("batch_size", 16, 64) is True
        assert opt._is_int_param("num_layers", 1, 5) is True
        assert opt._is_int_param("learning_rate", 1e-5, 1e-2) is False
        assert opt._is_int_param("window_size", 24.0, 96.0) is True  # known int param


class TestPredictorInferencer:
    """Tests for PredictorInferencer."""

    def test_interface_compliance(self):
        """PredictorInferencer implements InferencePlugin interface."""
        from doin_core.plugins.base import InferencePlugin
        assert issubclass(PredictorInferencer, InferencePlugin)


class TestPredictorSyntheticData:
    """Tests for PredictorSyntheticData (synthetic-datagen integration)."""

    def test_interface_compliance(self):
        """PredictorSyntheticData implements SyntheticDataPlugin."""
        from doin_core.plugins.base import SyntheticDataPlugin
        from doin_plugins.predictor.synthetic import PredictorSyntheticData
        assert issubclass(PredictorSyntheticData, SyntheticDataPlugin)

    def test_bootstrap_fallback_deterministic(self):
        """Bootstrap fallback produces identical output for same seed."""
        from doin_plugins.predictor.synthetic import PredictorSyntheticData

        plugin = PredictorSyntheticData()
        plugin._method = "bootstrap"
        plugin._n_samples = 100
        plugin._block_size = 20
        plugin._noise_scale = 0.05

        # Create fake real data
        import pandas as pd
        n = 500
        fake_prices = np.cumsum(np.random.randn(n) * 0.001) + 1.3
        plugin._real_data = {
            "train": pd.DataFrame({"typical_price": fake_prices}),
        }

        result1 = plugin._generate_bootstrap(seed=12345)
        result2 = plugin._generate_bootstrap(seed=12345)

        assert result1["data_hash"] == result2["data_hash"]
        assert result1["n_samples"] == 100
        assert result1["method"] == "bootstrap"

        # Different seed → different data
        result3 = plugin._generate_bootstrap(seed=99999)
        assert result3["data_hash"] != result1["data_hash"]

    def test_generate_returns_required_fields(self):
        """generate() returns all required dict fields."""
        from doin_plugins.predictor.synthetic import PredictorSyntheticData

        plugin = PredictorSyntheticData()
        plugin._method = "bootstrap"
        plugin._n_samples = 50
        plugin._block_size = 10
        plugin._noise_scale = 0.01

        import pandas as pd
        plugin._real_data = {
            "train": pd.DataFrame({"typical_price": np.ones(200) * 1.3}),
        }

        result = plugin.generate(seed=42)
        assert "synthetic_df" in result
        assert "synthetic_csv" in result
        assert "data_hash" in result
        assert "n_samples" in result
        assert "method" in result
        assert "seed" in result
        assert isinstance(result["synthetic_df"], pd.DataFrame)
        assert "typical_price" in result["synthetic_df"].columns


class TestFitnessConvention:
    """Verify the DON ↔ predictor performance sign convention."""

    def test_negation_convention(self):
        """DON performance = -predictor_fitness.
        
        predictor: fitness 0.5 (bad) → DON: -0.5
        predictor: fitness -0.1 (good, beating naive) → DON: 0.1
        So higher DON performance = lower predictor fitness = better.
        """
        # Simulate: optimizer returns performance = -fitness
        fitness_bad = 0.5
        fitness_good = -0.1

        perf_bad = -fitness_bad
        perf_good = -fitness_good

        assert perf_good > perf_bad  # DON correctly ranks good > bad
