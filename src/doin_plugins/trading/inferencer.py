"""DOIN evaluator adapter for trained agent-multi trading artifacts.

The DOIN ``InferencePlugin`` contract returns a scalar verification metric.
Rich action/intent serving remains a separate agent-multi or
prediction_provider API used by LTS; this adapter never trains and never
pretends that a scalar verification result is a live order instruction.
"""

from __future__ import annotations

import base64
import copy
import os
import tempfile
from pathlib import Path
from typing import Any

from doin_core.plugins.base import InferencePlugin

from .runtime import AgentMultiRuntime


class TradingInferencer(InferencePlugin):
    """Run agent-multi inference on requested/synthetic data and score it."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._runtime: AgentMultiRuntime | None = None

    def configure(self, config: dict[str, Any]) -> None:
        self._config = copy.deepcopy(config)
        self._runtime = AgentMultiRuntime(self._config)

    def evaluate(
        self,
        parameters: dict[str, Any],
        data: dict[str, Any] | None = None,
    ) -> float:
        if self._runtime is None:
            raise RuntimeError("TradingInferencer.configure() must be called first")

        run_config = copy.deepcopy(self._runtime.runtime_config)
        run_config.update(_clean_parameters(parameters))
        model_b64 = parameters.get("_model_b64")
        model_path = (
            parameters.get("_model_path")
            or parameters.get("model_path")
            or parameters.get("load_model")
            or run_config.get("load_model")
        )
        if not model_path and not model_b64:
            raise ValueError("trading inference requires a model artifact path")
        run_config["mode"] = "inference"
        run_config["quiet_mode"] = True

        temporary_data = None
        temporary_model = None
        try:
            if model_b64:
                temporary_model = tempfile.NamedTemporaryFile(
                    suffix=".zip", prefix="doin-trading-model-", delete=False
                )
                try:
                    temporary_model.write(base64.b64decode(model_b64, validate=True))
                except Exception as exc:
                    raise ValueError("invalid base64 trading model artifact") from exc
                finally:
                    temporary_model.close()
                run_config["load_model"] = temporary_model.name
            else:
                run_config["load_model"] = str(Path(model_path).expanduser())
            if data and data.get("synthetic_df") is not None:
                temporary_data = tempfile.NamedTemporaryFile(
                    suffix=".csv", prefix="doin-trading-scenario-", delete=False
                )
                temporary_data.close()
                data["synthetic_df"].to_csv(temporary_data.name, index=False)
                run_config["input_data_file"] = temporary_data.name
            summary = self._runtime.run(run_config, mode="inference")
            return _score_summary(summary, run_config)
        finally:
            if temporary_data is not None:
                try:
                    os.unlink(temporary_data.name)
                except OSError:
                    pass
            if temporary_model is not None:
                try:
                    os.unlink(temporary_model.name)
                except OSError:
                    pass


def _clean_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in parameters.items()
        if not key.startswith("_") and key not in {"model_path", "load_model"}
    }


def _score_summary(summary: dict[str, Any], config: dict[str, Any]) -> float:
    metric = str(
        config.get("inference_metric")
        or config.get("optimization_metric")
        or "total_return"
    ).lower()
    if metric in {"total_return", "return"}:
        return _required_metric(summary, "total_return")
    if metric in {"risk_adjusted_return", "rap"}:
        key = "rap" if metric == "rap" else "risk_adjusted_total_return"
        return _required_metric(summary, key)
    value = summary.get(metric)
    if value is None:
        raise KeyError(f"inference summary does not contain metric {metric!r}")
    return float(value)


def _required_metric(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key)
    if value is None:
        raise KeyError(f"inference summary does not contain metric {key!r}")
    return float(value)
