"""Evaluation metrics and reporting for the AI Workload Predictor."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics on held-out test data."""
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    latency_mae_ms: float
    latency_rmse_ms: float
    latency_r2: float
    confusion_matrix: List[List[int]]
    classes: List[str]
    classification_report_text: str = ""

    def summary(self) -> str:
        """Formats metrics into an ASCII summary report."""
        lines = [
            "================ Workload Predictor Evaluation ================",
            f"Accuracy            : {self.accuracy * 100.0:.2f}%",
            f"Macro Precision     : {self.precision_macro:.4f}",
            f"Macro Recall        : {self.recall_macro:.4f}",
            f"Macro F1-Score      : {self.f1_macro:.4f}",
            f"Weighted F1-Score   : {self.f1_weighted:.4f}",
            "----------------- Latency Regression (ms) ---------------------",
            f"MAE                 : {self.latency_mae_ms:.2f} ms",
            f"RMSE                : {self.latency_rmse_ms:.2f} ms",
            f"R^2 Score           : {self.latency_r2:.4f}",
            "------------------- Per-Class Report --------------------------",
            self.classification_report_text.strip(),
            "===============================================================",
        ]
        return "\n".join(lines)


def evaluate_workload_predictor(
    y_class_true: np.ndarray,
    y_class_pred: np.ndarray,
    y_lat_true: np.ndarray,
    y_lat_pred: np.ndarray,
    classes: Optional[List[str]] = None,
) -> EvaluationMetrics:
    """Computes comprehensive evaluation metrics on predictions against ground truth."""
    present_classes = sorted(list(set(y_class_true).union(set(y_class_pred))))
    labels = classes or present_classes

    acc = float(accuracy_score(y_class_true, y_class_pred))
    p_macro = float(precision_score(y_class_true, y_class_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_class_true, y_class_pred, average="macro", zero_division=0))
    f1_mac = float(f1_score(y_class_true, y_class_pred, average="macro", zero_division=0))

    p_weighted = float(precision_score(y_class_true, y_class_pred, average="weighted", zero_division=0))
    r_weighted = float(recall_score(y_class_true, y_class_pred, average="weighted", zero_division=0))
    f1_wei = float(f1_score(y_class_true, y_class_pred, average="weighted", zero_division=0))

    mae = float(mean_absolute_error(y_lat_true, y_lat_pred))
    rmse = float(np.sqrt(mean_squared_error(y_lat_true, y_lat_pred)))
    r2 = float(r2_score(y_lat_true, y_lat_pred))

    cm = confusion_matrix(y_class_true, y_class_pred, labels=labels).tolist()
    report = classification_report(y_class_true, y_class_pred, labels=labels, zero_division=0)

    return EvaluationMetrics(
        accuracy=acc,
        precision_macro=p_macro,
        recall_macro=r_macro,
        f1_macro=f1_mac,
        precision_weighted=p_weighted,
        recall_weighted=r_weighted,
        f1_weighted=f1_wei,
        latency_mae_ms=mae,
        latency_rmse_ms=rmse,
        latency_r2=r2,
        confusion_matrix=cm,
        classes=labels,
        classification_report_text=report,
    )
