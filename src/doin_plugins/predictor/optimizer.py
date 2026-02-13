"""DON Optimization Plugin for harveybc/predictor.

Wraps the predictor's existing plugin stack (predictor_plugin,
preprocessor_plugin, target_plugin) to perform one optimization step:

1. Load config + plugins from predictor's entry points
2. Resume from current best params (the DON network's champion)
3. Perturb hyperparameters within configured bounds
4. Train the model, compute fitness (lower = better → negated for DON)
5. Return (params, performance) — DON runner handles improvement detection

This replaces predictor's default_optimizer (DEAP GA) with DON's
decentralized optimization loop.  Zero changes to predictor code.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from doin_core.plugins.base import OptimizationPlugin


class PredictorOptimizer(OptimizationPlugin):
    """DON optimization plugin that drives predictor's model training.

    Config keys (all optional, sane defaults):
        predictor_root: Path to predictor repo (default: auto-detect)
        load_config: Path to predictor JSON config file
        predictor_plugin: Name of the predictor plugin (e.g. "mimo")
        preprocessor_plugin: Name of preprocessor plugin
        target_plugin: Name of target plugin
        hyperparameter_bounds: Dict of {param: [low, high]}
        step_size_fraction: Fraction of range for perturbation (default 0.15)
        seed: Random seed
    """

    # Parameters that must be rounded to int if bounds are both int
    _INT_HEURISTIC_PARAMS = {
        "num_layers", "layer_size", "early_patience", "batch_size",
        "encoder_conv_layers", "encoder_base_filters", "encoder_lstm_units",
        "horizon_attn_heads", "horizon_attn_key_dim", "horizon_embedding_dim",
        "kl_anneal_epochs", "window_size",
    }

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._predictor_config: dict[str, Any] = {}
        self._bounds: dict[str, tuple[float, float]] = {}
        self._step_frac = 0.15
        self._rng = random.Random()
        self._predictor_root: Path | None = None

        # Lazy-loaded predictor components
        self._predictor_plugin: Any = None
        self._preprocessor_plugin: Any = None
        self._target_plugin: Any = None

    def configure(self, config: dict[str, Any]) -> None:
        self._config = config
        self._step_frac = config.get("step_size_fraction", 0.15)

        seed = config.get("seed")
        if seed is not None:
            self._rng = random.Random(seed)
            np.random.seed(seed)

        # Resolve predictor repo root
        root = config.get("predictor_root")
        if root:
            self._predictor_root = Path(root).resolve()
        else:
            # Try common locations
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

        # Ensure predictor is importable
        root_str = str(self._predictor_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        # Load predictor's config (defaults + config file)
        self._load_predictor_config()

        # Extract hyperparameter bounds
        self._bounds = {}
        raw_bounds = self._predictor_config.get("hyperparameter_bounds", {})
        for key, val in raw_bounds.items():
            if isinstance(val, (list, tuple)) and len(val) == 2:
                self._bounds[key] = (float(val[0]), float(val[1]))

        # Load predictor plugins
        self._load_plugins()

    def _load_predictor_config(self) -> None:
        """Load predictor config: defaults → config file → DON overrides."""
        from app.config import DEFAULT_VALUES

        self._predictor_config = DEFAULT_VALUES.copy()

        # Load config file if specified
        config_file = self._config.get("load_config")
        if config_file:
            config_path = Path(config_file)
            if not config_path.is_absolute() and self._predictor_root:
                config_path = self._predictor_root / config_path
            if config_path.exists():
                with open(config_path) as f:
                    file_config = json.load(f)
                self._predictor_config.update(file_config)

        # Apply any DON-level overrides
        for key in [
            "predictor_plugin", "preprocessor_plugin", "target_plugin",
            "pipeline_plugin", "epochs", "batch_size", "window_size",
        ]:
            if key in self._config:
                self._predictor_config[key] = self._config[key]

        # Override hyperparameter bounds if provided at DON level
        if "hyperparameter_bounds" in self._config:
            self._predictor_config["hyperparameter_bounds"] = self._config["hyperparameter_bounds"]

        # Force optimizer-friendly settings
        self._predictor_config["disable_postfit_uncertainty"] = True

    def _load_plugins(self) -> None:
        """Load predictor's plugins via its entry point system."""
        from app.plugin_loader import load_plugin

        # Predictor plugin
        pred_name = self._predictor_config.get("predictor_plugin", "default_predictor")
        pred_cls, _ = load_plugin("predictor.plugins", pred_name)
        self._predictor_plugin = pred_cls(self._predictor_config)
        self._predictor_plugin.set_params(**self._predictor_config)

        # Preprocessor plugin
        pre_name = self._predictor_config.get("preprocessor_plugin", "default_preprocessor")
        pre_cls, _ = load_plugin("preprocessor.plugins", pre_name)
        self._preprocessor_plugin = pre_cls()
        self._preprocessor_plugin.set_params(**self._predictor_config)

        # Target plugin
        tgt_name = self._predictor_config.get("target_plugin", "default_target")
        tgt_cls, _ = load_plugin("target.plugins", tgt_name)
        self._target_plugin = tgt_cls()
        self._target_plugin.set_params(**self._predictor_config)

    def optimize(
        self,
        current_best_params: dict[str, Any] | None,
        current_best_performance: float | None,
    ) -> tuple[dict[str, Any], float]:
        """Run one optimization step.

        DON convention: higher performance = better.
        Predictor convention: lower fitness = better.
        We negate: DON performance = -predictor_fitness.
        """
        # Generate candidate hyperparameters
        if current_best_params is not None:
            candidate = self._perturb(current_best_params)
        else:
            candidate = self._random_params()

        # Evaluate candidate
        fitness = self._evaluate(candidate)

        # DON uses higher-is-better; predictor uses lower-is-better
        performance = -fitness

        return candidate, performance

    def _random_params(self) -> dict[str, Any]:
        """Generate random hyperparameters within bounds."""
        params: dict[str, Any] = {}
        for key, (low, high) in self._bounds.items():
            if self._is_int_param(key, low, high):
                params[key] = self._rng.randint(int(low), int(high))
            else:
                params[key] = self._rng.uniform(low, high)
        return params

    def _perturb(self, base_params: dict[str, Any]) -> dict[str, Any]:
        """Perturb parameters around base within bounds."""
        params: dict[str, Any] = {}
        for key, (low, high) in self._bounds.items():
            base_val = base_params.get(key)
            if base_val is None:
                # Parameter not in base — random init
                if self._is_int_param(key, low, high):
                    params[key] = self._rng.randint(int(low), int(high))
                else:
                    params[key] = self._rng.uniform(low, high)
                continue

            span = high - low
            sigma = span * self._step_frac
            new_val = float(base_val) + self._rng.gauss(0, sigma)
            new_val = max(low, min(high, new_val))  # clamp

            if self._is_int_param(key, low, high):
                params[key] = int(round(new_val))
            else:
                params[key] = new_val

        return params

    def _is_int_param(self, key: str, low: float, high: float) -> bool:
        """Heuristic: int if both bounds are int or key is known int param."""
        if key in self._INT_HEURISTIC_PARAMS:
            return True
        return isinstance(low, int) and isinstance(high, int)

    def _evaluate(self, hyper_params: dict[str, Any]) -> float:
        """Train model with given hyperparameters, return fitness (lower = better).

        This replicates the core eval logic from predictor's default_optimizer
        but for a single candidate, without DEAP.
        """
        import gc

        try:
            import tensorflow as tf
        except ImportError:
            tf = None

        # Clean up from any prior evaluation
        if hasattr(self._predictor_plugin, "model"):
            del self._predictor_plugin.model
        if tf:
            tf.keras.backend.clear_session()
        gc.collect()

        # Build evaluation config
        eval_config = copy.deepcopy(self._predictor_config)
        eval_config.update(hyper_params)
        eval_config["disable_postfit_uncertainty"] = True
        eval_config["mc_samples"] = 1

        # Handle special param types (matches predictor's convention)
        if "positional_encoding" in hyper_params:
            eval_config["positional_encoding"] = bool(int(round(hyper_params["positional_encoding"])))
        if "use_log1p_features" in hyper_params:
            val = hyper_params["use_log1p_features"]
            if not isinstance(val, list):
                val_int = int(round(val))
                eval_config["use_log1p_features"] = ["typical_price"] if val_int == 1 else None

        # Preprocess data
        datasets = self._preprocessor_plugin.run_preprocessing(
            self._target_plugin, eval_config
        )
        if isinstance(datasets, tuple):
            datasets = datasets[0]

        x_train = datasets["x_train"]
        y_train = datasets["y_train"]
        x_val = datasets["x_val"]
        y_val = datasets["y_val"]
        baseline_val = datasets.get("baseline_val")

        # Ensure 2D targets
        y_train = self._ensure_2d_targets(y_train)
        y_val = self._ensure_2d_targets(y_val)

        # Build model
        window_size = eval_config.get("window_size")
        plugin_name = eval_config.get("plugin", "ann")
        if plugin_name in ["lstm", "cnn", "transformer", "ann", "mimo", "n_beats", "tft"]:
            if len(x_train.shape) == 3:
                input_shape = (window_size, x_train.shape[2])
            else:
                input_shape = (x_train.shape[1],)
        else:
            input_shape = (x_train.shape[1],)

        self._predictor_plugin.set_params(**eval_config)
        self._predictor_plugin.build_model(
            input_shape=input_shape, x_train=x_train, config=eval_config
        )

        # Train
        history, train_preds, _, val_preds, _ = self._predictor_plugin.train(
            x_train, y_train,
            epochs=eval_config.get("epochs", 10),
            batch_size=eval_config.get("batch_size", 32),
            threshold_error=eval_config.get("threshold_error", 0.001),
            x_val=x_val, y_val=y_val, config=eval_config,
        )

        # Compute fitness (same as predictor's optimizer: denormalized MAE delta from naive)
        fitness = self._compute_fitness(
            val_preds, y_val, baseline_val, eval_config
        )

        return fitness

    def _compute_fitness(
        self,
        val_preds: list,
        y_val: Any,
        baseline_val: Any,
        config: dict[str, Any],
    ) -> float:
        """Compute fitness as avg delta from naive (lower = better).

        Matches predictor's default_optimizer fitness calculation.
        """
        try:
            from pipeline_plugins.stl_norm import denormalize, denormalize_returns
        except ImportError:
            # Fallback: use raw validation loss from training
            return float("inf")

        predicted_horizons = config.get("predicted_horizons", [1])
        max_horizon = max(predicted_horizons) if predicted_horizons else 1
        max_h_idx = predicted_horizons.index(max_horizon) if predicted_horizons else 0

        # Extract predictions for max horizon
        val_preds_h = np.asarray(val_preds[max_h_idx]).flatten()

        # Extract targets for max horizon
        if isinstance(y_val, dict):
            y_true = np.asarray(y_val[f"output_horizon_{max_horizon}"]).flatten()
        elif isinstance(y_val, list):
            y_true = np.asarray(y_val[max_h_idx]).flatten()
        else:
            y_true = np.asarray(y_val).flatten()

        # Align lengths
        n = min(len(val_preds_h), len(y_true))
        if baseline_val is not None:
            n = min(n, len(np.asarray(baseline_val).flatten()))
        if n <= 0:
            return float("inf")

        val_preds_h = val_preds_h[:n]
        y_true = y_true[:n]

        # Denormalize and compute MAE
        real_pred = denormalize(val_preds_h, config)
        real_true = denormalize(y_true, config)
        val_mae = float(np.mean(np.abs(real_pred - real_true)))

        # Naive MAE
        naive_mae = float("inf")
        if baseline_val is not None:
            baseline_h = np.asarray(baseline_val).flatten()[:n]
            real_baseline = denormalize(baseline_h, config)
            naive_mae = float(np.mean(np.abs(real_baseline - real_true)))

        # Fitness = delta from naive (how much worse than naive; lower = better)
        if np.isfinite(naive_mae) and naive_mae > 0:
            fitness = val_mae - naive_mae  # negative = beating naive
        else:
            fitness = val_mae

        return fitness

    @staticmethod
    def _ensure_2d_targets(y: Any) -> Any:
        """Match predictor's convention: targets as (N, 1) column vectors."""
        if isinstance(y, dict):
            return {
                k: np.asarray(v).reshape(-1, 1).astype(np.float32)
                for k, v in y.items()
            }
        return y

    def get_domain_metadata(self) -> dict[str, Any]:
        return {
            "performance_metric": "neg_fitness_delta_from_naive",
            "higher_is_better": True,
            "description": (
                "Timeseries prediction optimization via harveybc/predictor. "
                "Performance = -fitness where fitness is MAE delta from naive "
                "(lower fitness = better, so higher performance = better)."
            ),
            "predictor_plugin": self._predictor_config.get("predictor_plugin"),
            "predicted_horizons": self._predictor_config.get("predicted_horizons"),
        }
