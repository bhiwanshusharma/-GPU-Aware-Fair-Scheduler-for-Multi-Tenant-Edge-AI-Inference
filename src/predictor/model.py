"""Lightweight edge-suitable machine learning model for workload classification and latency estimation."""

import os
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.core.types import WorkloadClass, PredictionOutput
from src.core.exceptions import ModelNotFoundError, PredictorUnavailableError
from src.predictor.features import FEATURE_NAMES, JobFeatures


class LightweightWorkloadPredictor:
    """Edge-optimized dual predictor:
    1. Classifier: WorkloadClass (GPU_INTENSIVE, CPU_BOUND, IO_BOUND, MIXED).
    2. Regressor: Estimated execution latency in ms.
    
    Trained model artifact footprint is <250KB with microsecond-level edge inference.
    """

    def __init__(
        self,
        classifier: Optional[RandomForestClassifier] = None,
        regressor: Optional[RandomForestRegressor] = None,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.feature_names = feature_names or list(FEATURE_NAMES)
        self.classifier = classifier or RandomForestClassifier(
            n_estimators=30,
            max_depth=6,
            random_state=42,
            n_jobs=1,  # Predictable single-thread CPU execution for edge
        )
        self.regressor = regressor or RandomForestRegressor(
            n_estimators=20,
            max_depth=5,
            random_state=42,
            n_jobs=1,
        )
        self.metadata = metadata or {}
        self.is_trained = False

    def fit(
        self,
        X: np.ndarray,
        y_class: np.ndarray,
        y_latency: np.ndarray,
    ) -> "LightweightWorkloadPredictor":
        """Fits both classification and regression models on training data."""
        self.classifier.fit(X, y_class)
        self.regressor.fit(X, y_latency)
        self.is_trained = True
        self.metadata["trained_at"] = time.time()
        self.metadata["num_training_samples"] = len(X)
        self.metadata["classes"] = list(self.classifier.classes_)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts classification labels, confidence scores, and latencies.
        
        Args:
            X: Input feature array of shape (N, num_features).
            
        Returns:
            Tuple of:
            - pred_classes: Array of predicted category strings of shape (N,)
            - confidences: Array of prediction confidence floats [0, 1] of shape (N,)
            - pred_latencies: Array of estimated execution latencies in ms of shape (N,)
        """
        if not self.is_trained or self.classifier is None or self.regressor is None:
            raise PredictorUnavailableError("Model has not been trained yet.")

        # Class predictions and confidence probabilities
        probs = self.classifier.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        pred_classes = self.classifier.classes_[pred_idx]
        confidences = np.max(probs, axis=1)

        # Latency predictions (clamped to non-negative)
        pred_latencies = np.maximum(0.1, self.regressor.predict(X))

        return pred_classes, confidences, pred_latencies

    def predict_single(self, features: JobFeatures) -> PredictionOutput:
        """Runs low-latency inference on a single JobFeatures instance."""
        X = features.to_array().reshape(1, -1)
        classes, confs, lats = self.predict(X)

        raw_class_str = str(classes[0])
        try:
            wl_class = WorkloadClass(raw_class_str)
        except ValueError:
            wl_class = WorkloadClass.MIXED

        # Estimate memory footprint
        est_mem = float(features.memory_footprint_mb)

        return PredictionOutput(
            estimated_latency_ms=float(lats[0]),
            estimated_memory_mb=est_mem,
            classification=wl_class,
            confidence_score=float(confs[0]),
        )

    def save(self, file_path: str) -> str:
        """Serializes the trained models and metadata to a single file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        payload = {
            "classifier": self.classifier,
            "regressor": self.regressor,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "is_trained": self.is_trained,
        }
        joblib.dump(payload, file_path, compress=3)
        return file_path

    @classmethod
    def load(cls, file_path: str) -> "LightweightWorkloadPredictor":
        """Deserializes and constructs a predictor instance from disk."""
        if not os.path.exists(file_path):
            raise ModelNotFoundError(f"Trained model artifact not found: {file_path}")

        try:
            payload = joblib.load(file_path)
            if not isinstance(payload, dict) or "classifier" not in payload or "regressor" not in payload:
                raise ModelNotFoundError(f"Invalid model artifact structure in: {file_path}")
            predictor = cls(
                classifier=payload["classifier"],
                regressor=payload["regressor"],
                feature_names=payload.get("feature_names", list(FEATURE_NAMES)),
                metadata=payload.get("metadata", {}),
            )
            predictor.is_trained = payload.get("is_trained", True)
            return predictor
        except Exception as e:
            if isinstance(e, ModelNotFoundError):
                raise
            raise ModelNotFoundError(f"Failed to load model artifact '{file_path}': {e}")
