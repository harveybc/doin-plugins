"""External DOIN optimizer adapter for agent-multi trading experiments.

``TradingOptimizer`` is intentionally analogous to the established predictor
adapter.  The local DEAP/NEAT or other optimizer remains in agent-multi and is
usable through its own CLI.  This class only adds DOIN migration callbacks and
the ``OptimizationPlugin`` contract.
"""

from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any, Callable

from doin_core.plugins.base import OptimizationPlugin

from .runtime import AgentMultiRuntime


class TradingOptimizer(OptimizationPlugin):
    """Wrap agent-multi's local optimizer without modifying doin-node."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._runtime: AgentMultiRuntime | None = None
        self._network_champion: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._callbacks: dict[str, Callable[..., Any] | None] = {}
        self._force_stage_advance = threading.Event()
        self._statistics_cache: dict[str, Any] = {}

    def configure(self, config: dict[str, Any]) -> None:
        self._config = copy.deepcopy(config)
        self._runtime = AgentMultiRuntime(self._config)

    def optimize(
        self,
        current_best_params: dict[str, Any] | None,
        current_best_performance: float | None,
    ) -> tuple[dict[str, Any], float]:
        if self._runtime is None:
            raise RuntimeError("TradingOptimizer.configure() must be called first")

        seed_params = current_best_params or self._take_network_champion()
        run_config = copy.deepcopy(self._runtime.runtime_config)
        run_config.update({
            key: value for key, value in self._config.items()
            if key not in {
                "agent_multi_root", "base_config", "load_config", "agent_multi_config"
            }
        })
        # Keep candidate evaluation deterministic across nodes while giving
        # each optimization island a distinct search trajectory. Training and
        # evaluation seeds remain canonical; only the local GA seed is offset.
        run_config["ga_seed"] = int(run_config.get("ga_seed", 0)) + int(
            run_config.get("node_seed_offset", 0)
        )
        if seed_params:
            run_config["initial_candidate_params"] = _clean_parameters(seed_params)
        run_config["current_best_performance"] = current_best_performance
        run_config["optimization_callbacks"] = self._build_callbacks()
        # Distributed verification requires the exact trained champion, not a
        # fresh retraining from the same hyperparameters.
        run_config["optimization_capture_model_artifact"] = True
        run_config["optimization_require_model_artifact"] = True

        env, agent, pipeline, _ = self._runtime.build_components(run_config)
        local_optimizer = self._runtime.load_local_optimizer(run_config)
        result = local_optimizer.optimize(
            env_plugin=env,
            agent_plugin=agent,
            pipeline_plugin=pipeline,
            config=run_config,
        )
        if not isinstance(result, dict):
            raise TypeError("agent-multi optimizer must return a parameter dictionary")

        performance = result.get("_best_fitness")
        if performance is None:
            raise ValueError(
                "agent-multi optimizer did not return '_best_fitness'; "
                "the DOIN adapter cannot report an unscored champion"
            )
        parameters = {
            key: value for key, value in result.items()
            if not key.startswith("_")
        }
        parameters.update({
            "_doin_adapter": "trading_agent_multi_v1",
            "_agent_multi_config_hash": self._runtime.config_hash,
            "_metric_schema": run_config.get("metric_schema", "trading.metrics.v1"),
        })
        model_b64 = result.get("_best_model_b64")
        if model_b64:
            parameters["_model_b64"] = model_b64
        best_metrics = result.get("_best_metrics")
        if isinstance(best_metrics, dict):
            parameters["_metric_evidence"] = best_metrics
        return parameters, float(performance)

    def get_domain_metadata(self) -> dict[str, Any]:
        return {
            "performance_metric": self._config.get(
                "optimization_metric", "agent_multi_fitness"
            ),
            "metric_schema": self._config.get("metric_schema", "trading.metrics.v1"),
            "higher_is_better": bool(self._config.get("higher_is_better", True)),
            "domain_type": "agent-multi-trading",
            "config_hash": self._runtime.config_hash if self._runtime else None,
        }

    def get_runtime_statistics(self) -> dict[str, Any]:
        """Return durable local optimizer counters without scanning on every poll."""
        if self._runtime is None:
            return {"candidate_evaluations_total": 0}

        raw_path = self._runtime.runtime_config.get("optimization_candidate_history")
        if not raw_path:
            return {"candidate_evaluations_total": 0}
        path = Path(os.path.expandvars(str(raw_path))).expanduser()
        if not path.is_absolute():
            path = self._runtime.root / path
        path = path.resolve()

        try:
            stat = path.stat()
        except OSError:
            return {
                "candidate_evaluations_total": 0,
                "candidate_history_source": str(path),
            }

        cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
        if self._statistics_cache.get("key") != cache_key:
            with path.open("rb") as handle:
                row_count = sum(1 for _line in handle)
            self._statistics_cache = {
                "key": cache_key,
                "candidate_evaluations_total": max(0, row_count - 1),
                "candidate_history_source": str(path),
            }
        return {
            key: value for key, value in self._statistics_cache.items()
            if key != "key"
        }

    # DOIN island-model callback surface. These names intentionally match the
    # existing predictor adapter and doin-node callback wiring.
    def set_network_champion(self, params: dict[str, Any]) -> None:
        with self._lock:
            self._network_champion = copy.deepcopy(params)

    def _take_network_champion(self) -> dict[str, Any] | None:
        with self._lock:
            champion = self._network_champion
            self._network_champion = None
            return champion

    def _set_callback(self, name: str, callback: Callable[..., Any]) -> None:
        self._callbacks[name] = callback

    def set_local_champion_callback(self, callback: Callable[..., Any]) -> None:
        self._set_callback("on_new_champion", callback)

    def set_eval_service_callback(self, callback: Callable[..., Any]) -> None:
        self._set_callback("on_between_candidates", callback)

    def set_generation_end_callback(self, callback: Callable[..., Any]) -> None:
        self._set_callback("on_generation_end", callback)

    def set_stage_start_callback(self, callback: Callable[..., Any]) -> None:
        self._set_callback("on_stage_start", callback)

    def set_stage_end_callback(self, callback: Callable[..., Any]) -> None:
        self._set_callback("on_stage_end", callback)

    def set_candidate_evaluated_callback(self, callback: Callable[..., Any]) -> None:
        self._set_callback("on_candidate_evaluated", callback)

    def force_stage_advance(self) -> None:
        self._force_stage_advance.set()

    def _build_callbacks(self) -> dict[str, Callable[..., Any]]:
        callbacks = {
            name: callback for name, callback in self._callbacks.items() if callback
        }
        if self._force_stage_advance.is_set():
            self._force_stage_advance.clear()
        callbacks["network_champion_provider"] = self._take_network_champion
        callbacks["stage_advance_requested"] = self._force_stage_advance.is_set
        return callbacks


def _clean_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in parameters.items()
        if not key.startswith("_doin_") and not key.startswith("_metric_")
    }
