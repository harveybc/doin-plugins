"""DOIN Optimization Plugin for harveybc/predictor.

Wraps predictor's real DEAP GA optimizer (default_optimizer) with
DOIN island-model callbacks for:
  - Migration IN: inject network champion into population before each generation
  - Migration OUT: broadcast new local champion to network
  - Eval service: process 1 pending evaluation request between candidates
  - Stats: report generation-level metrics to DOIN dashboard/OLAP

Each DOIN node runs the FULL predictor optimization (DEAP GA with
incremental stages, populations, generations, crossover, mutation).
DOIN only adds the migration operator — champion sharing between nodes.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from doin_core.plugins.base import OptimizationPlugin

# Activation mapping (must match predictor's default_optimizer)
ACTIVATIONS = ["relu", "elu", "selu", "tanh", "sigmoid", "swish", "gelu", "leaky_relu"]


class PredictorOptimizer(OptimizationPlugin):
    """DOIN optimization plugin that wraps predictor's DEAP GA optimizer.

    This does NOT replace the GA — it wraps it, adding island-model
    migration via DOIN callbacks.

    Config keys (all optional, sane defaults):
        predictor_root: Path to predictor repo (default: auto-detect)
        load_config: Path to predictor JSON config file
        predictor_plugin: Name of the predictor plugin (e.g. "mimo")
        preprocessor_plugin: Name of preprocessor plugin
        target_plugin: Name of target plugin
        hyperparameter_bounds: Dict of {param: [low, high]}
        epochs: Training epochs per candidate
        batch_size: Batch size for training
        + all predictor config keys (passed through)
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._predictor_config: dict[str, Any] = {}
        self._predictor_root: Path | None = None
        self._predictor_plugin: Any = None
        self._preprocessor_plugin: Any = None
        self._target_plugin: Any = None
        self._deap_optimizer: Any = None  # predictor's default_optimizer.Plugin

        # Thread-safe state for DOIN communication
        self._network_champion: dict[str, Any] | None = None  # Set by unified node
        self._network_champion_lock = threading.Lock()
        self._local_champion_callback: Callable | None = None  # Set by unified node
        self._eval_service_callback: Callable | None = None  # Set by unified node
        self._generation_end_callback: Callable | None = None  # Set by unified node

        # Metrics exposed to DOIN
        self._current_generation: int = 0
        self._current_stage: int = 1
        self._total_stages: int = 1
        self._total_candidates_evaluated: int = 0
        self._last_gen_stats: dict[str, Any] = {}
        self._is_running: bool = False

    def configure(self, config: dict[str, Any]) -> None:
        self._config = config

        # Resolve predictor repo
        root = config.get("predictor_root")
        if root:
            self._predictor_root = Path(root).resolve()
        else:
            for candidate in [
                Path.home() / "predictor",
                Path.cwd() / "predictor",
                Path("/home/openclaw/predictor"),
                Path("/home/openclaw/.openclaw/workspace/predictor"),
            ]:
                if (candidate / "setup.py").exists():
                    self._predictor_root = candidate
                    break

        if self._predictor_root is None:
            raise FileNotFoundError(
                "Cannot find predictor repo. Set 'predictor_root' in config."
            )

        root_str = str(self._predictor_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        self._load_predictor_config()
        self._load_plugins()
        self._load_deap_optimizer()

    def _load_predictor_config(self) -> None:
        from app.config import DEFAULT_VALUES

        self._predictor_config = DEFAULT_VALUES.copy()

        config_file = self._config.get("load_config")
        if config_file:
            config_path = Path(config_file)
            if not config_path.is_absolute() and self._predictor_root:
                config_path = self._predictor_root / config_path
            if config_path.exists():
                with open(config_path) as f:
                    self._predictor_config.update(json.load(f))

        # Override with DOIN config values
        for key, val in self._config.items():
            if key not in ("predictor_root", "load_config", "optimization_callbacks"):
                self._predictor_config[key] = val

        self._predictor_config["disable_postfit_uncertainty"] = True
        self._predictor_config.setdefault("quiet", True)

    def _load_plugins(self) -> None:
        from app.plugin_loader import load_plugin

        pred_name = self._predictor_config.get("predictor_plugin", "default_predictor")
        pred_cls, _ = load_plugin("predictor.plugins", pred_name)
        self._predictor_plugin = pred_cls(self._predictor_config)
        self._predictor_plugin.set_params(**self._predictor_config)

        pre_name = self._predictor_config.get("preprocessor_plugin", "default_preprocessor")
        pre_cls, _ = load_plugin("preprocessor.plugins", pre_name)
        self._preprocessor_plugin = pre_cls()
        self._preprocessor_plugin.set_params(**self._predictor_config)

        tgt_name = self._predictor_config.get("target_plugin", "default_target")
        tgt_cls, _ = load_plugin("target.plugins", tgt_name)
        self._target_plugin = tgt_cls()
        self._target_plugin.set_params(**self._predictor_config)

    def _load_deap_optimizer(self) -> None:
        """Load predictor's real DEAP GA optimizer."""
        from optimizer_plugins.default_optimizer import Plugin as DeapOptimizer
        self._deap_optimizer = DeapOptimizer()
        self._deap_optimizer.set_params(**self._predictor_config)

    # ── DOIN Integration Points ──────────────────────────────

    def set_network_champion(self, params: dict[str, Any]) -> None:
        """Called by unified node when a network champion is received (migration IN)."""
        with self._network_champion_lock:
            self._network_champion = params

    def set_local_champion_callback(self, callback: Callable) -> None:
        """Set callback for when a new local champion is found (migration OUT).
        callback(params_dict, fitness, metrics_dict, generation, stage_info)
        """
        self._local_champion_callback = callback

    def set_eval_service_callback(self, callback: Callable) -> None:
        """Set callback to process one pending evaluation between candidates.
        callback(gen, candidate_num, stage_info) -> None
        """
        self._eval_service_callback = callback

    def set_generation_end_callback(self, callback: Callable) -> None:
        """Set callback for end-of-generation stats reporting.
        callback(population, hof, hyper_keys, gen, stage_info, stats) -> None
        """
        self._generation_end_callback = callback

    # ── Callback Implementations ─────────────────────────────

    def _on_generation_start(self, population, hof, hyper_keys, gen, stage_info):
        """Migration IN: return network champion params to inject into population."""
        self._current_generation = gen
        self._current_stage = stage_info.get("stage", 1)
        self._total_stages = stage_info.get("total_stages", 1)
        self._total_candidates_evaluated = stage_info.get("total_candidates_evaluated", 0)

        with self._network_champion_lock:
            champion = self._network_champion
            self._network_champion = None  # Consume it

        if champion is not None:
            # Return raw params — the predictor callback hook will inject into population
            return champion
        return None

    def _on_new_champion(self, champion_params, fitness, metrics, gen, stage_info):
        """Migration OUT: broadcast new local champion to DOIN network."""
        if self._local_champion_callback:
            self._local_champion_callback(champion_params, fitness, metrics, gen, stage_info)

    def _on_between_candidates(self, gen, candidate_num, stage_info):
        """Process one pending evaluation request between candidates."""
        self._total_candidates_evaluated = stage_info.get("total_candidates_evaluated", 0)
        if self._eval_service_callback:
            self._eval_service_callback(gen, candidate_num, stage_info)

    def _on_generation_end(self, population, hof, hyper_keys, gen, stage_info, stats):
        """Report generation-level stats."""
        self._last_gen_stats = stage_info
        if self._generation_end_callback:
            self._generation_end_callback(population, hof, hyper_keys, gen, stage_info, stats)

    # ── Main Optimization Entry Point ────────────────────────

    def optimize(
        self,
        current_best_params: dict[str, Any] | None,
        current_best_performance: float | None,
    ) -> tuple[dict[str, Any], float] | None:
        """Run the FULL predictor DEAP GA optimization with DOIN callbacks.

        This runs the entire GA (all stages, all generations) — NOT one step.
        Returns the best hyperparameters found and their performance.
        """
        self._is_running = True

        # Suppress TF noise
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("PREDICTOR_QUIET", "1")

        # Build config for predictor optimizer
        opt_config = copy.deepcopy(self._predictor_config)

        # Inject DOIN callbacks
        opt_config["optimization_callbacks"] = {
            "on_generation_start": self._on_generation_start,
            "on_new_champion": self._on_new_champion,
            "on_between_candidates": self._on_between_candidates,
            "on_generation_end": self._on_generation_end,
        }

        # If we have a network champion, inject it as the starting point
        if current_best_params:
            # Write champion params to a temp file so predictor can load it
            params_file = opt_config.get("optimization_parameters_file", "optimization_parameters.json")
            try:
                with open(params_file, "w") as f:
                    json.dump(current_best_params, f)
                opt_config["optimization_resume"] = True
                opt_config["optimization_parameters_file"] = params_file
            except Exception:
                pass

        try:
            # Run the FULL DEAP GA optimization
            best_hyper = self._deap_optimizer.optimize(
                self._predictor_plugin,
                self._preprocessor_plugin,
                opt_config,
            )
        except Exception as e:
            print(f"[DOIN] DEAP optimization error: {e}")
            import traceback
            traceback.print_exc()
            self._is_running = False
            return None
        finally:
            self._is_running = False

        if best_hyper is None:
            return None

        # Get final champion fitness from the optimizer
        fitness = getattr(self._deap_optimizer, "best_fitness_so_far", float("inf"))
        performance = -fitness  # DOIN convention: higher = better

        return best_hyper, performance

    # ── Metrics Properties ───────────────────────────────────

    def get_domain_metadata(self) -> dict[str, Any]:
        """Return metadata about the predictor optimization domain."""
        return {
            "performance_metric": "fitness",
            "higher_is_better": False,  # predictor: lower fitness = better
            "domain_type": "predictor-timeseries",
            "optimizer": "DEAP GA with incremental stages",
        }

    @property
    def last_round_metrics(self) -> dict[str, Any]:
        """Detailed metrics from the optimization."""
        optimizer = self._deap_optimizer
        if optimizer is None:
            return {}
        return {
            "generation": self._current_generation,
            "stage": self._current_stage,
            "total_stages": self._total_stages,
            "total_candidates_evaluated": self._total_candidates_evaluated,
            "champion_fitness": getattr(optimizer, "best_fitness_so_far", None),
            "train_mae": getattr(optimizer, "best_train_mae_so_far", None),
            "train_naive_mae": getattr(optimizer, "best_train_naive_mae_so_far", None),
            "val_mae": getattr(optimizer, "best_val_mae_so_far", None),
            "val_naive_mae": getattr(optimizer, "best_naive_mae_so_far", None),
            "test_mae": getattr(optimizer, "best_test_mae_so_far", None),
            "test_naive_mae": getattr(optimizer, "best_test_naive_mae_so_far", None),
            "is_running": self._is_running,
        }
