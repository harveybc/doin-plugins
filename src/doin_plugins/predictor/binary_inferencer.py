"""DON Inference Plugin for binary classification predictor models.

Subclass of PredictorInferencer that evaluates binary classification
performance (AUC-ROC) instead of regression MAE for model verification.

Used by DON evaluators to verify optimizer-reported performance on
synthetic data using binary classification metrics.
"""

from __future__ import annotations

import copy
import gc
import os
import tempfile
from typing import Any

import numpy as np

from doin_plugins.predictor.inferencer import PredictorInferencer


class BinaryPredictorInferencer(PredictorInferencer):
    """Evaluates binary predictor model performance for DON verification.

    Overrides _compute_fitness to use AUC-ROC instead of MAE + denormalize.
    """

    def _compute_fitness(
        self,
        val_preds: list,
        y_val: Any,
        baseline_val: Any,
        config: dict[str, Any],
    ) -> float:
        """Binary classification fitness (lower is better).

        Computes AUC-ROC on validation predictions and returns -AUC
        so that lower values indicate better performance (matching
        the existing DON convention).
        """
        predicted_horizons = config.get("predicted_horizons", [1])
        max_h_idx = 0  # Binary always horizon 1

        val_preds_h = np.asarray(val_preds[max_h_idx]).flatten()

        if isinstance(y_val, dict):
            key = list(y_val.keys())[0]
            y_true = np.asarray(y_val[key]).flatten()
        elif isinstance(y_val, list):
            y_true = np.asarray(y_val[max_h_idx]).flatten()
        else:
            y_true = np.asarray(y_val).flatten()

        n = min(len(val_preds_h), len(y_true))
        if n <= 0:
            return float("inf")

        val_preds_h = val_preds_h[:n]
        y_true = y_true[:n]

        # Import binary fitness from predictor
        try:
            from predictor_plugins.common.binary_fitness import (
                compute_binary_metrics_for_split,
                compute_binary_val_only_fitness,
                find_best_threshold,
            )
            best_threshold = find_best_threshold(y_true, val_preds_h)
            val_metrics = compute_binary_metrics_for_split(y_true, val_preds_h, threshold=best_threshold)
            return compute_binary_val_only_fitness(val_metrics)
        except ImportError:
            # Fallback: compute AUC directly
            try:
                from sklearn.metrics import roc_auc_score
                if len(np.unique(y_true.astype(int))) < 2:
                    return 0.0  # Single class — can't compute AUC
                auc = float(roc_auc_score(y_true.astype(int), val_preds_h))
                return -auc  # Lower is better
            except Exception:
                return float("inf")
