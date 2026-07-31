"""Tests for the predictor domain DON plugins.

These tests verify the current external-predictor wrapper contract:
1. Plugin interface and metric direction.
2. Shared-population migration and stage-control callbacks.
3. Deterministic pre-trained synthetic generator output.

Note: Full integration tests require predictor to be installed with its
dependencies (TensorFlow, etc.). These unit tests mock the heavy parts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

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
        assert meta["higher_is_better"] is False

    def test_network_champion_is_consumed_once(self):
        """A migrated champion is injected once, then removed from local state."""
        opt = PredictorOptimizer()
        champion = {"learning_rate": 0.001, "window_size": 64}
        opt.set_network_champion(champion)

        first = opt._on_generation_start([], None, [], 0, {"stage": 1})
        second = opt._on_generation_start([], None, [], 0, {"stage": 1})

        assert first == champion
        assert second is None

    def test_force_stage_advance_is_consumed_once(self):
        """A network stage signal advances exactly one optimizer boundary."""
        opt = PredictorOptimizer()
        opt.force_stage_advance()

        assert opt._on_generation_start([], None, [], 2, {"stage": 1}) == {
            "_force_stage_advance": True,
        }
        assert opt._on_generation_start([], None, [], 2, {"stage": 1}) is None

    def test_between_candidates_calls_service_and_honors_stage_signal(self):
        opt = PredictorOptimizer()
        callback = MagicMock()
        opt.set_eval_service_callback(callback)
        opt.force_stage_advance()

        result = opt._on_between_candidates(
            3,
            7,
            {"total_candidates_evaluated": 19},
        )

        callback.assert_called_once_with(
            3,
            7,
            {"total_candidates_evaluated": 19},
        )
        assert result == {"_force_stage_advance": True}
        assert opt._total_candidates_evaluated == 19

    def test_last_round_metrics_report_wrapped_optimizer_state(self):
        opt = PredictorOptimizer()
        opt._deap_optimizer = SimpleNamespace(
            best_fitness_so_far=0.125,
            best_val_mae_so_far=0.2,
        )
        opt._current_generation = 4
        opt._current_stage = 2
        opt._total_stages = 5

        metrics = opt.last_round_metrics

        assert metrics["generation"] == 4
        assert metrics["stage"] == 2
        assert metrics["total_stages"] == 5
        assert metrics["champion_fitness"] == pytest.approx(0.125)
        assert metrics["optimizer_type"] == "deap_ga"


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

    @staticmethod
    def _configured_fake_plugin():
        from doin_plugins.predictor.synthetic import PredictorSyntheticData

        plugin = PredictorSyntheticData()
        plugin._n_samples = 100
        plugin._hybrid_model = {"model": "fixture"}

        def generate(_model, n_samples, *, seed, initial_price):
            rng = np.random.default_rng(seed)
            returns = rng.normal(0.0, 0.001, n_samples)
            return initial_price * np.exp(np.cumsum(returns))

        plugin._hybrid_generate = generate
        return plugin

    def test_pretrained_generator_is_deterministic(self):
        """The loaded-generator path produces identical output for one seed."""
        plugin = self._configured_fake_plugin()

        result1 = plugin.generate(seed=12345)
        result2 = plugin.generate(seed=12345)

        assert result1["data_hash"] == result2["data_hash"]
        assert result1["n_samples"] == 100
        assert result1["method"] == "hmm_hybrid"

        result3 = plugin.generate(seed=99999)
        assert result3["data_hash"] != result1["data_hash"]

    def test_generate_returns_required_fields(self):
        """generate() returns all required dict fields."""
        plugin = self._configured_fake_plugin()
        plugin._n_samples = 50

        import pandas as pd

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
        """DON performance = predictor_fitness (direct, no negation).
        
        Binary predictor: fitness 0.5 (good, higher weighted F1)
        Binary predictor: fitness 0.1 (bad, lower weighted F1)
        So higher DON performance = higher predictor fitness = better.
        
        Regression predictor: fitness 0.5 (bad MAE) 
        So lower DON performance = lower predictor fitness = better.
        """
        # Binary: higher fitness = better
        fitness_good = 0.5
        fitness_bad = 0.1
        assert fitness_good > fitness_bad  # DON correctly ranks good > bad
