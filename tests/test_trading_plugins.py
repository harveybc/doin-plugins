from __future__ import annotations

import base64
import threading
from pathlib import Path
from typing import Any

import pytest

from doin_core.plugins.base import InferencePlugin, OptimizationPlugin, SyntheticDataPlugin
from doin_plugins.trading import (
    TradingInferencer,
    TradingOptimizer,
    TradingScenarioSyntheticData,
)
from doin_plugins.trading.runtime import _copy_runtime_overrides


def test_trading_entry_plugins_implement_existing_contracts() -> None:
    assert issubclass(TradingOptimizer, OptimizationPlugin)
    assert issubclass(TradingInferencer, InferencePlugin)
    assert issubclass(TradingScenarioSyntheticData, SyntheticDataPlugin)


def test_runtime_override_copy_preserves_bound_callbacks() -> None:
    class CallbackOwner:
        def __init__(self) -> None:
            self.lock = threading.Lock()

        def callback(self) -> None:
            return None

    owner = CallbackOwner()
    source = {
        "nested": {"values": [1, 2]},
        "optimization_callbacks": {"on_new_champion": owner.callback},
    }

    copied = _copy_runtime_overrides(source)

    assert copied["nested"] == source["nested"]
    assert copied["nested"] is not source["nested"]
    assert copied["optimization_callbacks"] is not source["optimization_callbacks"]
    assert copied["optimization_callbacks"]["on_new_champion"] == owner.callback


def test_scenario_fixture_is_deterministic_and_aligned() -> None:
    plugin = TradingScenarioSyntheticData()
    plugin.configure({"assets": ["SOLUSDT", "EURUSD"], "scenario_rows": 8})

    first, first_hash = plugin.generate_with_hash(seed=42)
    second, second_hash = plugin.generate_with_hash(seed=42)

    assert first_hash == second_hash
    assert first["data_hash"] == second["data_hash"]
    assert first["synthetic_csv"] == second["synthetic_csv"]
    assert first["synthetic_df"].shape == (16, 7)
    assert set(first["synthetic_df"]["asset"]) == {"SOLUSDT", "EURUSD"}


def test_scenario_fixture_rejects_unpromoted_backend() -> None:
    plugin = TradingScenarioSyntheticData()
    with pytest.raises(ValueError, match="scenario_backend"):
        plugin.configure({"scenario_backend": "learned_generator_v1"})


def test_trading_optimizer_metadata_is_explicit() -> None:
    plugin = TradingOptimizer()
    plugin._config = {
        "optimization_metric": "mean_weekly_rap",
        "metric_schema": "trading.metrics.v1",
        "higher_is_better": True,
    }
    metadata = plugin.get_domain_metadata()
    assert metadata["performance_metric"] == "mean_weekly_rap"
    assert metadata["metric_schema"] == "trading.metrics.v1"
    assert metadata["higher_is_better"] is True


def test_doin_optimizer_delegates_to_local_optimizer(monkeypatch) -> None:
    import doin_plugins.trading.optimizer as trading_optimizer

    class FakeLocalOptimizer:
        def optimize(self, **kwargs):
            assert kwargs["config"]["optimization_capture_model_artifact"] is True
            assert kwargs["config"]["optimization_require_model_artifact"] is True
            assert kwargs["config"]["ga_seed"] == 12
            callbacks = kwargs["config"]["optimization_callbacks"]
            callbacks["on_new_champion"](
                {"learning_rate": 0.1}, 0.25, {"mean_weekly_rap": 0.25}, 0,
                {"stage": 1, "total_stages": 1},
            )
            return {
                "learning_rate": 0.1,
                "_best_fitness": 0.25,
                "_best_model_b64": base64.b64encode(b"model").decode("ascii"),
                "_best_metrics": {"mean_weekly_rap": 0.25},
            }

    class FakeRuntime:
        config_hash = "sha256:test"
        runtime_config = {
            "env_plugin": "fake_env",
            "agent_plugin": "fake_agent",
            "pipeline_plugin": "fake_pipeline",
            "optimizer_plugin": "fake_optimizer",
        }

        def __init__(self, config):
            self.doin_config = config

        def build_components(self, overrides=None):
            return object(), object(), object(), dict(self.runtime_config)

        def load_local_optimizer(self, config):
            return FakeLocalOptimizer()

    monkeypatch.setattr(trading_optimizer, "AgentMultiRuntime", FakeRuntime)
    plugin = trading_optimizer.TradingOptimizer()
    plugin.configure({
        "optimization_metric": "mean_weekly_rap",
        "ga_seed": 10,
        "node_seed_offset": 2,
    })
    seen = []
    plugin.set_local_champion_callback(lambda *args: seen.append(args))

    params, performance = plugin.optimize({"learning_rate": 0.2}, 0.1)

    assert params["learning_rate"] == 0.1
    assert params["_doin_adapter"] == "trading_agent_multi_v1"
    assert base64.b64decode(params["_model_b64"]) == b"model"
    assert params["_metric_evidence"]["mean_weekly_rap"] == pytest.approx(0.25)
    assert performance == pytest.approx(0.25)
    assert seen and seen[0][1] == pytest.approx(0.25)


def test_trading_optimizer_delegates_shared_population_bridge(monkeypatch) -> None:
    import doin_plugins.trading.optimizer as trading_optimizer

    class FakeLocalOptimizer:
        def __init__(self) -> None:
            self.setup_config = None

        def setup_shared_mode(self, **kwargs):
            self.setup_config = kwargs["config"]

        def create_shared_population(self, population_size, *, seed):
            assert population_size == 4
            assert seed == 71
            return {"population": [{"parameters": {"score": 0.1}}]}

        def evaluate_candidate(self, genome, generation):
            assert genome == {"parameters": {"score": 0.1}}
            assert generation == 3
            return {"fitness": 0.2, "hyper_dict": {"score": 0.1}}

        def reproduce_shared(self, *args):
            return {"generation": args[1] + 1, "population": []}

    local = FakeLocalOptimizer()

    class FakeRuntime:
        config_hash = "sha256:test"
        runtime_config = {
            "env_plugin": "fake_env",
            "agent_plugin": "fake_agent",
            "pipeline_plugin": "fake_pipeline",
            "optimizer_plugin": "fake_optimizer",
            "ga_seed": 10,
        }

        def __init__(self, config):
            self.doin_config = config

        def build_components(self, overrides=None):
            return object(), object(), object(), dict(self.runtime_config)

        def load_local_optimizer(self, config):
            return local

    monkeypatch.setattr(trading_optimizer, "AgentMultiRuntime", FakeRuntime)
    plugin = trading_optimizer.TradingOptimizer()
    plugin.configure({"ga_seed": 10, "node_seed_offset": 2})

    state = plugin.create_shared_population(4, seed=71)
    result = plugin.evaluate_candidate(state["population"][0], generation=3)
    next_state = plugin.reproduce_shared(
        state["population"], 3, 99, {}, [], {}, 0, 0,
    )

    assert local.setup_config["ga_seed"] == 10
    assert local.setup_config["optimization_capture_model_artifact"] is True
    assert local.setup_config["optimization_require_model_artifact"] is True
    assert result["fitness"] == pytest.approx(0.2)
    assert next_state["generation"] == 4


def test_local_optimizer_accepts_seed_and_emits_doin_callbacks() -> None:
    from optimizer_plugins.default_optimizer import Plugin

    class FakeAgent:
        def hparam_schema(self):
            return [("score", 0.0, 1.0, "float")]

        def set_params(self, **kwargs: Any) -> None:
            self.params = kwargs

        def fitness(self, summary, config):
            return float(summary["total_return"])

    class FakeEnv:
        def close(self) -> None:
            return None

    class FakePipeline:
        def run_pipeline(self, *, config, env_plugin, agent_plugin, mode):
            return {"total_return": float(config["score"]), "mode": mode}

    champions = []
    candidates = []
    optimizer = Plugin()
    result = optimizer.optimize(
        env_plugin=FakeEnv(),
        agent_plugin=FakeAgent(),
        pipeline_plugin=FakePipeline(),
        config={
            "ga_population": 3,
            "ga_generations": 1,
            "ga_mutpb": 0.0,
            "ga_cxpb": 0.0,
            "ga_seed": 2,
            "optimization_patience": 2,
            "initial_candidate_params": {"score": 0.99},
            "optimization_callbacks": {
                "on_new_champion": lambda *args: champions.append(args),
                "on_candidate_evaluated": lambda info: candidates.append(info),
            },
        },
    )

    assert result["_best_fitness"] == pytest.approx(0.99)
    assert champions
    assert candidates


def test_trading_optimizer_reports_durable_candidate_history(tmp_path) -> None:
    history = tmp_path / "candidate_history.csv"
    history.write_text(
        "timestamp_utc,fitness\n"
        "2026-07-13T12:00:00+00:00,0.1\n"
        "2026-07-13T12:05:00+00:00,0.2\n"
        "2026-07-13T12:10:00+00:00,0.3\n",
        encoding="utf-8",
    )
    plugin = TradingOptimizer()
    plugin._runtime = type("Runtime", (), {
        "root": tmp_path,
        "runtime_config": {"optimization_candidate_history": str(history)},
    })()

    first = plugin.get_runtime_statistics()
    assert first["candidate_evaluations_total"] == 3
    assert first["candidate_history_source"] == str(history)
    assert first["candidates_per_hour"] == pytest.approx(12.0)
    assert first["candidate_seconds_median"] == pytest.approx(300.0)
    assert first["rate_sample_size"] == 2

    with history.open("a", encoding="utf-8") as handle:
        handle.write("2026-07-13T12:15:00+00:00,0.4\n")
    assert plugin.get_runtime_statistics()["candidate_evaluations_total"] == 4


def test_trading_optimizer_rate_excludes_restart_downtime(tmp_path) -> None:
    history = tmp_path / "candidate_history.csv"
    history.write_text(
        "timestamp_utc,fitness\n"
        "2026-07-12T08:00:00+00:00,0.1\n"
        "2026-07-13T12:00:00+00:00,0.2\n"
        "2026-07-13T12:06:00+00:00,0.3\n"
        "2026-07-13T12:12:00+00:00,0.4\n",
        encoding="utf-8",
    )
    plugin = TradingOptimizer()
    plugin._runtime = type("Runtime", (), {
        "root": tmp_path,
        "runtime_config": {"optimization_candidate_history": str(history)},
    })()

    statistics = plugin.get_runtime_statistics()

    assert statistics["candidate_evaluations_total"] == 4
    assert statistics["candidates_per_hour"] == pytest.approx(10.0)
    assert statistics["rate_sample_size"] == 2


def test_trading_inferencer_decodes_champion_and_uses_metric_plugin_output(monkeypatch) -> None:
    import doin_plugins.trading.inferencer as trading_inferencer

    seen: dict[str, Any] = {}

    class FakeRuntime:
        runtime_config = {
            "optimization_metric": "risk_adjusted_return",
            "metrics_plugin": "trading_metrics",
        }

        def __init__(self, config):
            self.config = config

        def run(self, overrides=None, *, mode):
            seen.update(overrides or {})
            seen["mode"] = mode
            model_path = Path(seen["load_model"])
            assert model_path.read_bytes() == b"verified-model"
            return {
                "total_return": 0.4,
                "max_drawdown_fraction": 0.9,
                "risk_adjusted_total_return": 0.17,
                "rap": 0.17,
            }

    monkeypatch.setattr(trading_inferencer, "AgentMultiRuntime", FakeRuntime)
    plugin = trading_inferencer.TradingInferencer()
    plugin.configure({"optimization_metric": "risk_adjusted_return"})

    score = plugin.evaluate({
        "_model_b64": base64.b64encode(b"verified-model").decode("ascii"),
    })

    assert score == pytest.approx(0.17)
    assert seen["mode"] == "inference"
