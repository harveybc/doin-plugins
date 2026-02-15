"""Tests for the reference quadratic plugins."""

import numpy as np
import pytest

from doin_plugins.quadratic_optimizer import QuadraticOptimizer
from doin_plugins.quadratic_inferencer import QuadraticInferencer
from doin_plugins.quadratic_synthetic import QuadraticSyntheticData


TARGET = [1.0, 2.0, 3.0, 4.0, 5.0]
PLUGIN_CONFIG = {"n_params": 5, "target": TARGET, "step_size": 0.5, "seed": 42}


class TestQuadraticOptimizer:
    def test_first_step(self) -> None:
        opt = QuadraticOptimizer()
        opt.configure(PLUGIN_CONFIG)
        params, perf = opt.optimize(None, None)
        assert "x" in params
        assert len(params["x"]) == 5
        assert perf <= 0  # Negative MSE

    def test_improves_over_steps(self) -> None:
        opt = QuadraticOptimizer()
        opt.configure({**PLUGIN_CONFIG, "seed": 123})
        params, perf = opt.optimize(None, None)

        # Run many steps — should converge toward target
        for _ in range(200):
            new_params, new_perf = opt.optimize(params, perf)
            if new_perf > perf:
                params, perf = new_params, new_perf

        # Should be significantly better than initial
        assert perf > -10.0  # Close to 0 = close to target


class TestQuadraticInferencer:
    def test_evaluate_exact_match(self) -> None:
        inf = QuadraticInferencer()
        inf.configure({"target": TARGET})
        perf = inf.evaluate({"x": TARGET})
        assert perf == pytest.approx(0.0)  # Perfect match

    def test_evaluate_with_error(self) -> None:
        inf = QuadraticInferencer()
        inf.configure({"target": TARGET})
        bad_x = [0.0, 0.0, 0.0, 0.0, 0.0]
        perf = inf.evaluate({"x": bad_x})
        assert perf < 0  # Negative MSE

    def test_evaluate_with_synthetic_data(self) -> None:
        inf = QuadraticInferencer()
        inf.configure({})
        # Pass target through data (synthetic data path)
        perf = inf.evaluate({"x": TARGET}, data={"target": TARGET})
        assert perf == pytest.approx(0.0)

    def test_no_target_raises(self) -> None:
        inf = QuadraticInferencer()
        inf.configure({})
        with pytest.raises(ValueError, match="No target"):
            inf.evaluate({"x": [1, 2, 3]})


class TestQuadraticSyntheticData:
    def test_generate(self) -> None:
        gen = QuadraticSyntheticData()
        gen.configure({"target": TARGET, "noise_std": 0.1})
        data = gen.generate(seed=42)
        assert "target" in data
        assert data["synthetic"] is True
        assert len(data["target"]) == len(TARGET)

    def test_noisy_but_close(self) -> None:
        gen = QuadraticSyntheticData()
        gen.configure({"target": TARGET, "noise_std": 0.01})
        data = gen.generate(seed=42)
        # Should be close but not exact
        diff = np.abs(np.array(data["target"]) - np.array(TARGET))
        assert np.all(diff < 0.1)  # Very close with low noise

    def test_no_target_generates_from_seed(self) -> None:
        """Without explicit target, generates deterministic data from seed."""
        gen = QuadraticSyntheticData()
        gen.configure({"n_params": 5})
        data = gen.generate(seed=42)
        assert len(data["target"]) == 5
        assert data["synthetic"] is True
        # Same seed → same data
        data2 = gen.generate(seed=42)
        assert data["target"] == data2["target"]
        # Different seed → different data
        data3 = gen.generate(seed=99)
        assert data["target"] != data3["target"]
