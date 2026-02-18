"""DON Inference Plugin for harveybc/predictor.

Used by DON evaluators to verify optimizer-reported performance.
Loads the predictor model with given hyperparameters, trains it,
and independently computes the fitness to confirm the claimed metric.
"""

from __future__ import annotations

import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from doin_core.plugins.base import InferencePlugin


class PredictorInferencer(InferencePlugin):
    """Evaluates predictor model performance for DON verification.

    Same config structure as PredictorOptimizer — both need access
    to the predictor repo and its plugin stack.
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._predictor_config: dict[str, Any] = {}
        self._predictor_root: Path | None = None
        self._predictor_plugin: Any = None
        self._preprocessor_plugin: Any = None
        self._target_plugin: Any = None

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

        for key in [
            "predictor_plugin", "preprocessor_plugin", "target_plugin",
            "epochs", "batch_size", "window_size",
        ]:
            if key in self._config:
                self._predictor_config[key] = self._config[key]

        self._predictor_config["disable_postfit_uncertainty"] = True
        # Quiet mode: suppress verbose predictor output by default
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

    def evaluate(
        self,
        parameters: dict[str, Any],
        data: dict[str, Any] | None = None,
    ) -> float:
        """Evaluate model with given hyperparameters.

        Trains from scratch with the given params and computes the
        fitness metric independently to verify optimizer claims.

        Args:
            parameters: Hyperparameters from the optimae.
            data: Optional synthetic data dict. If None, uses configured
                  train/val files from predictor config.

        Returns:
            Performance metric (negated fitness — higher = better).
        """
        import builtins
        import os

        # Suppress noisy predictor prints in quiet mode
        if self._predictor_config.get("quiet", True):
            os.environ["PREDICTOR_QUIET"] = "1"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            _orig_print = builtins.print
            def _quiet_print(*args, **kwargs):
                if args:
                    msg = str(args[0]).upper()
                    if any(k in msg for k in ["ERROR", "WARN", "EXCEPTION", "FATAL"]):
                        _orig_print(*args, **kwargs)
            builtins.print = _quiet_print

        try:
            import tensorflow as tf
        except ImportError:
            tf = None

        # Clean slate
        if hasattr(self._predictor_plugin, "model"):
            del self._predictor_plugin.model
        if tf:
            tf.keras.backend.clear_session()
        gc.collect()

        # Build eval config with the claimed hyperparams
        eval_config = copy.deepcopy(self._predictor_config)
        eval_config.update(parameters)
        eval_config["disable_postfit_uncertainty"] = True
        eval_config["mc_samples"] = 1

        # Special param handling (must match optimizer's conventions)
        ACTIVATIONS = ["relu", "elu", "selu", "tanh", "sigmoid", "swish", "gelu", "linear"]
        if "activation" in parameters:
            idx = int(round(parameters["activation"])) % len(ACTIVATIONS)
            eval_config["activation"] = ACTIVATIONS[idx]
        if "positional_encoding" in parameters:
            eval_config["positional_encoding"] = bool(int(round(parameters["positional_encoding"])))
        if "use_log1p_features" in parameters:
            val = parameters["use_log1p_features"]
            if not isinstance(val, list):
                eval_config["use_log1p_features"] = (
                    ["typical_price"] if int(round(val)) == 1 else None
                )
        # Ensure integer params are ints
        for int_param in ["window_size", "encoder_conv_layers", "encoder_base_filters",
                          "encoder_lstm_units", "horizon_attn_heads", "horizon_attn_key_dim",
                          "horizon_embedding_dim", "batch_size", "early_patience", "kl_anneal_epochs"]:
            if int_param in eval_config and isinstance(eval_config[int_param], float):
                eval_config[int_param] = int(round(eval_config[int_param]))

        # Preprocess — use synthetic data if provided, else standard files
        if data is not None and "synthetic_df" in data:
            # Synthetic data from DOIN evaluator verification.
            # The synthetic plugin generates a complete price series.
            # We preprocess it through the same pipeline the optimizer uses,
            # splitting into train/val so the model can be trained and
            # its generalization independently verified.
            import tempfile, os
            synth_df = data["synthetic_df"]
            n = len(synth_df)
            # Same split ratios as real data pipeline (60/20/20)
            i_train = int(n * 0.6)
            i_val = int(n * 0.8)
            train_df = synth_df.iloc[:i_train]
            val_df = synth_df.iloc[i_train:i_val]
            test_df = synth_df.iloc[i_val:]
            with tempfile.TemporaryDirectory() as tmpdir:
                train_path = os.path.join(tmpdir, "train.csv")
                val_path = os.path.join(tmpdir, "val.csv")
                test_path = os.path.join(tmpdir, "test.csv")
                train_df.to_csv(train_path, index=False)
                val_df.to_csv(val_path, index=False)
                test_df.to_csv(test_path, index=False)
                synth_config = copy.deepcopy(eval_config)
                synth_config["training_file"] = train_path
                synth_config["validation_file"] = val_path
                synth_config["testing_file"] = test_path
                datasets = self._preprocessor_plugin.run_preprocessing(
                    self._target_plugin, synth_config
                )
                if isinstance(datasets, tuple):
                    datasets = datasets[0]
        elif data is not None:
            datasets = data
        else:
            datasets = self._preprocessor_plugin.run_preprocessing(
                self._target_plugin, eval_config
            )
            if isinstance(datasets, tuple):
                datasets = datasets[0]

        x_train = datasets["x_train"]
        y_train = self._ensure_2d(datasets["y_train"])
        x_val = datasets["x_val"]
        y_val = self._ensure_2d(datasets["y_val"])
        baseline_val = datasets.get("baseline_val")

        # Build & train
        window_size = eval_config.get("window_size")
        plugin_name = eval_config.get("plugin", "ann")
        if plugin_name in ["lstm", "cnn", "transformer", "ann", "mimo", "n_beats", "tft"]:
            input_shape = (
                (window_size, x_train.shape[2]) if len(x_train.shape) == 3
                else (x_train.shape[1],)
            )
        else:
            input_shape = (x_train.shape[1],)

        self._predictor_plugin.set_params(**eval_config)
        self._predictor_plugin.build_model(
            input_shape=input_shape, x_train=x_train, config=eval_config,
        )

        history, train_preds, _, val_preds, _ = self._predictor_plugin.train(
            x_train, y_train,
            epochs=eval_config.get("epochs", 10),
            batch_size=eval_config.get("batch_size", 32),
            threshold_error=eval_config.get("threshold_error", 0.001),
            x_val=x_val, y_val=y_val, config=eval_config,
        )

        # Compute fitness
        fitness = self._compute_fitness(val_preds, y_val, baseline_val, eval_config)

        # Return DON performance (higher = better)
        return -fitness

    def _compute_fitness(
        self,
        val_preds: list,
        y_val: Any,
        baseline_val: Any,
        config: dict[str, Any],
    ) -> float:
        """Same fitness calculation as PredictorOptimizer."""
        try:
            from pipeline_plugins.stl_norm import denormalize
        except ImportError:
            return float("inf")

        predicted_horizons = config.get("predicted_horizons", [1])
        max_horizon = max(predicted_horizons) if predicted_horizons else 1
        max_h_idx = predicted_horizons.index(max_horizon) if predicted_horizons else 0

        val_preds_h = np.asarray(val_preds[max_h_idx]).flatten()

        if isinstance(y_val, dict):
            y_true = np.asarray(y_val[f"output_horizon_{max_horizon}"]).flatten()
        elif isinstance(y_val, list):
            y_true = np.asarray(y_val[max_h_idx]).flatten()
        else:
            y_true = np.asarray(y_val).flatten()

        n = min(len(val_preds_h), len(y_true))
        if baseline_val is not None:
            n = min(n, len(np.asarray(baseline_val).flatten()))
        if n <= 0:
            return float("inf")

        val_preds_h = val_preds_h[:n]
        y_true = y_true[:n]

        real_pred = denormalize(val_preds_h, config)
        real_true = denormalize(y_true, config)
        val_mae = float(np.mean(np.abs(real_pred - real_true)))

        naive_mae = float("inf")
        if baseline_val is not None:
            baseline_h = np.asarray(baseline_val).flatten()[:n]
            real_baseline = denormalize(baseline_h, config)
            naive_mae = float(np.mean(np.abs(real_baseline - real_true)))

        if np.isfinite(naive_mae) and naive_mae > 0:
            return val_mae - naive_mae
        return val_mae

    @staticmethod
    def _ensure_2d(y: Any) -> Any:
        if isinstance(y, dict):
            return {
                k: np.asarray(v).reshape(-1, 1).astype(np.float32)
                for k, v in y.items()
            }
        return y
