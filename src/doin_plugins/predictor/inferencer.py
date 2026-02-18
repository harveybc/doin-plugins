"""DON Inference Plugin for harveybc/predictor.

Used by DON evaluators to verify optimizer-reported performance.
Loads the trained model shared by the optimizer, generates synthetic data,
and runs INFERENCE ONLY (no training) to verify the claimed metric.
"""

from __future__ import annotations

import base64
import copy
import gc
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from doin_core.plugins.base import InferencePlugin


class PredictorInferencer(InferencePlugin):
    """Evaluates predictor model performance for DON verification.

    The optimizer shares its trained model (base64-encoded .keras file)
    along with the hyperparameters. The evaluator:
    1. Generates synthetic data (unique per-evaluator seed)
    2. Preprocesses it through the same pipeline
    3. Loads the trained model
    4. Runs model.predict() (inference only — NO training)
    5. Computes fitness metric independently
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._predictor_config: dict[str, Any] = {}
        self._predictor_root: Path | None = None
        self._preprocessor_plugin: Any = None
        self._target_plugin: Any = None

    def configure(self, config: dict[str, Any]) -> None:
        self._config = config

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
        self._predictor_config.setdefault("quiet", True)

    def _load_plugins(self) -> None:
        from app.plugin_loader import load_plugin

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
        """Evaluate model via INFERENCE ONLY using the optimizer's trained model.

        The optimizer includes '_model_b64' (base64-encoded .keras file) in
        the parameters. We decode it, load the model, run predict() on
        synthetic data, and compute fitness.

        Args:
            parameters: Hyperparameters + _model_b64 from the optimae.
            data: Synthetic data dict from DOIN evaluator verification.

        Returns:
            Performance metric (negated fitness — higher = better).
        """
        import builtins

        if self._predictor_config.get("quiet", True):
            os.environ["PREDICTOR_QUIET"] = "1"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        try:
            import tensorflow as tf
        except ImportError:
            tf = None

        # Extract trained model
        model_b64 = parameters.pop("_model_b64", None)
        if model_b64 is None:
            # No model provided — can't verify without it
            return float("-inf")  # Worst possible score

        # Build eval config with claimed hyperparams
        eval_config = copy.deepcopy(self._predictor_config)
        # Don't put non-config keys in eval_config
        clean_params = {k: v for k, v in parameters.items() if not k.startswith("_")}
        eval_config.update(clean_params)
        eval_config["disable_postfit_uncertainty"] = True
        eval_config["mc_samples"] = 1

        # Special param handling (must match optimizer's conventions)
        ACTIVATIONS = ["relu", "elu", "selu", "tanh", "sigmoid", "swish", "gelu", "linear"]
        if "activation" in clean_params:
            idx = int(round(clean_params["activation"])) % len(ACTIVATIONS)
            eval_config["activation"] = ACTIVATIONS[idx]
        if "positional_encoding" in clean_params:
            eval_config["positional_encoding"] = bool(int(round(clean_params["positional_encoding"])))
        for int_param in ["window_size", "encoder_conv_layers", "encoder_base_filters",
                          "encoder_lstm_units", "horizon_attn_heads", "horizon_attn_key_dim",
                          "horizon_embedding_dim", "batch_size", "early_patience", "kl_anneal_epochs"]:
            if int_param in eval_config and isinstance(eval_config[int_param], float):
                eval_config[int_param] = int(round(eval_config[int_param]))

        # Preprocess synthetic data for inference
        if data is not None and "synthetic_df" in data:
            synth_df = data["synthetic_df"]
            n = len(synth_df)
            i_train = int(n * 0.6)
            i_val = int(n * 0.8)
            train_df = synth_df.iloc[:i_train]
            val_df = synth_df.iloc[i_train:i_val]
            test_df = synth_df.iloc[i_val:]
            with tempfile.TemporaryDirectory() as tmpdir:
                for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                    df.to_csv(os.path.join(tmpdir, f"{name}.csv"), index=False)
                synth_config = copy.deepcopy(eval_config)
                synth_config["training_file"] = os.path.join(tmpdir, "train.csv")
                synth_config["validation_file"] = os.path.join(tmpdir, "val.csv")
                synth_config["testing_file"] = os.path.join(tmpdir, "test.csv")
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

        x_val = datasets["x_val"]
        y_val = self._ensure_2d(datasets["y_val"])
        baseline_val = datasets.get("baseline_val")

        # Load the trained model from base64 — INFERENCE ONLY
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(base64.b64decode(model_b64))
            tmp_model_path = tmp.name

        try:
            model = tf.keras.models.load_model(tmp_model_path, compile=False)

            # Run inference
            pred_bs = int(eval_config.get("batch_size", 32))
            val_preds = model.predict(x_val, batch_size=pred_bs, verbose=0)
            val_preds = [val_preds] if isinstance(val_preds, np.ndarray) else val_preds

            # Compute fitness
            fitness = self._compute_fitness(val_preds, y_val, baseline_val, eval_config)
        finally:
            # Cleanup
            os.unlink(tmp_model_path)
            if tf:
                tf.keras.backend.clear_session()
            gc.collect()

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
