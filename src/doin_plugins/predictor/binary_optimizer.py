"""DOIN Optimization Plugin for binary classification predictor models.

Subclass of PredictorOptimizer that exposes binary classification metrics
(AUC-ROC, F1, Accuracy, MCC) instead of regression MAE metrics.

The underlying NEAT/DEAP optimizer is identical — the binary candidate_worker
in the predictor repo already computes binary metrics when
target_plugin=="binary_target". This plugin simply:
1. Overrides get_domain_metadata() to declare binary metric semantics.
2. Overrides last_round_metrics to expose binary-specific metric names.
"""

from __future__ import annotations

from typing import Any

from doin_plugins.predictor.optimizer import PredictorOptimizer


class BinaryPredictorOptimizer(PredictorOptimizer):
    """DOIN optimization plugin for binary classification predictors.

    Identical to PredictorOptimizer but reports binary metrics
    (AUC-ROC, F1) instead of MAE. The actual metric computation
    happens in predictor's candidate_worker (binary branch).
    """

    def get_domain_metadata(self) -> dict[str, Any]:
        """Return metadata about the binary predictor optimization domain."""
        return {
            "performance_metric": "binary_fitness",
            "metric_type": "binary",
            "higher_is_better": True,  # Binary fitness: higher = better (positive weighted F1)
            "domain_type": "predictor-binary-classification",
            "optimizer": "NEAT GA with staged optimization",
            "primary_metric": "Accuracy",
            "secondary_metric": "F1",
            "metric_labels": {
                "val_mae": "Val Accuracy",
                "train_mae": "Train Accuracy",
                "val_naive_mae": "Val F1",
                "train_naive_mae": "Train F1",
                "test_mae": "Test Accuracy",
                "test_naive_mae": "Test F1",
                "fitness": "Binary Fitness (weighted F1)",
            },
        }

    @property
    def last_round_metrics(self) -> dict[str, Any]:
        """Detailed metrics from the optimization with binary labels."""
        metrics = super().last_round_metrics

        # Add binary metadata so dashboard can display correct labels
        metrics["metric_type"] = "binary"
        metrics["metric_labels"] = {
            "val_mae": "Val AUC-ROC",
            "train_mae": "Train AUC-ROC",
            "val_naive_mae": "Val F1",
            "train_naive_mae": "Train F1",
            "test_mae": "Test AUC-ROC",
            "test_naive_mae": "Test F1",
        }

        # Rename for clarity in the metrics dict (keep originals for compatibility)
        metrics["val_auc_roc"] = metrics.get("val_mae")
        metrics["train_auc_roc"] = metrics.get("train_mae")
        metrics["val_f1"] = metrics.get("val_naive_mae")
        metrics["train_f1"] = metrics.get("train_naive_mae")
        metrics["test_auc_roc"] = metrics.get("test_mae")
        metrics["test_f1"] = metrics.get("test_naive_mae")

        return metrics
