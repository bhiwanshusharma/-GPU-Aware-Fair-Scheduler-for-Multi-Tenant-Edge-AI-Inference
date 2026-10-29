"""Workload profiling, feature extraction, and AI prediction modules."""

from src.predictor.base import BasePredictor
from src.predictor.features import WorkloadFeatureExtractor, JobFeatures, FEATURE_NAMES
from src.predictor.model import LightweightWorkloadPredictor
from src.predictor.inference import WorkloadInferenceEngine
from src.predictor.dataset import (
    generate_synthetic_workload_dataset,
    save_dataset_to_csv,
    load_dataset_from_csv,
    train_test_split_dataset,
)
from src.predictor.trainer import train_workload_predictor
from src.predictor.evaluator import evaluate_workload_predictor, EvaluationMetrics

# Convenient alias
WorkloadPredictor = WorkloadInferenceEngine

__all__ = [
    "BasePredictor",
    "WorkloadPredictor",
    "WorkloadFeatureExtractor",
    "JobFeatures",
    "FEATURE_NAMES",
    "LightweightWorkloadPredictor",
    "WorkloadInferenceEngine",
    "generate_synthetic_workload_dataset",
    "save_dataset_to_csv",
    "load_dataset_from_csv",
    "train_test_split_dataset",
    "train_workload_predictor",
    "evaluate_workload_predictor",
    "EvaluationMetrics",
]
