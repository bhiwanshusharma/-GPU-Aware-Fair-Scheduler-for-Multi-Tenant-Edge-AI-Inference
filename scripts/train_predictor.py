#!/usr/bin/env python3
"""Training script for the AI Workload Predictor."""

import argparse
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predictor.trainer import train_workload_predictor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the edge AI workload prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/workload_dataset.csv",
        help="Path to CSV dataset (generated automatically if not existing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/workload_predictor.joblib",
        help="Target file path to save serialized model artifact",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of dataset reserved for held-out testing (0.0 to 1.0)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1200,
        help="Number of synthetic samples to generate if creating fresh dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic pseudo-random seed",
    )

    args = parser.parse_args()

    print(f"[*] Starting AI Workload Predictor training pipeline...")
    print(f"    Dataset path : {args.dataset}")
    print(f"    Output model : {args.output}")
    print(f"    Test split   : {args.test_size * 100:.1f}%")
    print(f"    Random seed  : {args.seed}")

    predictor, metrics = train_workload_predictor(
        dataset_path=args.dataset,
        output_model_path=args.output,
        test_size=args.test_size,
        random_seed=args.seed,
        num_samples=args.samples,
    )

    print("\n" + metrics.summary() + "\n")
    print(f"[+] Model artifact successfully generated at: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
