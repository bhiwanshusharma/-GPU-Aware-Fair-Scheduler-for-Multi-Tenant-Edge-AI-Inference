"""Training pipeline for the AI Workload Predictor."""

import os
import logging
from typing import Tuple, Optional

from src.predictor.dataset import (
    generate_synthetic_workload_dataset,
    save_dataset_to_csv,
    load_dataset_from_csv,
    train_test_split_dataset,
)
from src.predictor.model import LightweightWorkloadPredictor
from src.predictor.evaluator import evaluate_workload_predictor, EvaluationMetrics

logger = logging.getLogger(__name__)


def train_workload_predictor(
    dataset_path: Optional[str] = None,
    output_model_path: str = "models/workload_predictor.joblib",
    test_size: float = 0.2,
    random_seed: int = 42,
    num_samples: int = 1200,
) -> Tuple[LightweightWorkloadPredictor, EvaluationMetrics]:
    """Executes the end-to-end model training pipeline with zero data leakage.
    
    1. Loads existing CSV dataset or generates synthetic edge workload dataset.
    2. Performs stratified train/test split.
    3. Fits lightweight classification and latency regression models on train set.
    4. Evaluates performance strictly on held-out test set.
    5. Serializes trained model artifact to disk.
    
    Returns:
        Tuple of (trained_predictor, evaluation_metrics).
    """
    # 1. Dataset Acquisition
    if dataset_path and os.path.exists(dataset_path):
        logger.info(f"Loading existing dataset from {dataset_path}...")
        X, y_class, y_latency = load_dataset_from_csv(dataset_path)
    else:
        logger.info("Generating synthetic edge AI workload dataset...")
        X, y_class, y_latency = generate_synthetic_workload_dataset(
            num_samples=num_samples,
            random_seed=random_seed,
        )
        saved_csv = dataset_path or "data/workload_dataset.csv"
        save_dataset_to_csv(X, y_class, y_latency, saved_csv)
        logger.info(f"Saved dataset to {saved_csv}")

    # 2. Train / Test Split (Strict separation to prevent data leakage)
    X_train, X_test, y_train, y_test, y_lat_train, y_lat_test = train_test_split_dataset(
        X, y_class, y_latency, test_size=test_size, random_seed=random_seed
    )
    logger.info(f"Split data into Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # 3. Model Training
    predictor = LightweightWorkloadPredictor()
    predictor.fit(X_train, y_train, y_lat_train)

    # 4. Evaluation on Held-Out Test Set
    pred_classes, confs, pred_lats = predictor.predict(X_test)
    metrics = evaluate_workload_predictor(
        y_class_true=y_test,
        y_class_pred=pred_classes,
        y_lat_true=y_lat_test,
        y_lat_pred=pred_lats,
    )

    predictor.metadata["metrics"] = {
        "accuracy": metrics.accuracy,
        "f1_macro": metrics.f1_macro,
        "latency_mae_ms": metrics.latency_mae_ms,
    }

    # 5. Model Serialization
    predictor.save(output_model_path)
    logger.info(f"Model successfully saved to {output_model_path}")

    return predictor, metrics
