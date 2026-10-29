#!/usr/bin/env python3
"""Evaluation script for the AI Workload Predictor."""

import argparse
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predictor.model import LightweightWorkloadPredictor
from src.predictor.dataset import load_dataset_from_csv, train_test_split_dataset
from src.predictor.evaluator import evaluate_workload_predictor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the serialized AI workload prediction model on test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/workload_predictor.joblib",
        help="Path to serialized model artifact (.joblib)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/workload_dataset.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test partition ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[-] Error: Model file not found at {args.model}. Train the model first using scripts/train_predictor.py.")
        return 1

    if not os.path.exists(args.dataset):
        print(f"[-] Error: Dataset file not found at {args.dataset}.")
        return 1

    print(f"[*] Loading model from {args.model}...")
    predictor = LightweightWorkloadPredictor.load(args.model)

    print(f"[*] Loading dataset from {args.dataset}...")
    X, y_class, y_latency = load_dataset_from_csv(args.dataset)

    # Use identical test split
    _, X_test, _, y_class_test, _, y_lat_test = train_test_split_dataset(
        X, y_class, y_latency, test_size=args.test_size, random_seed=args.seed
    )

    print(f"[*] Evaluating on {len(X_test)} held-out test samples...")
    pred_classes, confs, pred_lats = predictor.predict(X_test)

    metrics = evaluate_workload_predictor(
        y_class_true=y_class_test,
        y_class_pred=pred_classes,
        y_lat_true=y_lat_test,
        y_lat_pred=pred_lats,
    )

    print("\n" + metrics.summary() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
