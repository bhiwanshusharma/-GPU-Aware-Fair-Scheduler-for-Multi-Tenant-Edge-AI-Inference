"""Dataset generation, loading, and train/test splitting for workload prediction."""

import os
import csv
import random
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

from src.core.types import WorkloadClass
from src.predictor.features import FEATURE_NAMES


WORKLOAD_CLASSES = [
    WorkloadClass.GPU_INTENSIVE.value,
    WorkloadClass.CPU_BOUND.value,
    WorkloadClass.IO_BOUND.value,
    WorkloadClass.MIXED.value,
]


def generate_synthetic_workload_dataset(
    num_samples: int = 1200,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a physically calibrated synthetic edge AI dataset.
    
    Generates samples across 4 distinct operational categories:
    - GPU_INTENSIVE: High GPU utilization, high FLOPs, high batch size
    - CPU_BOUND: High CPU utilization, low GPU utilization, light compute
    - IO_BOUND: High I/O wait, low CPU/GPU, streaming packet ingest
    - MIXED: Balanced compute, moderate memory, MobileNet-scale networks
    
    Returns:
        X: Feature matrix of shape (num_samples, len(FEATURE_NAMES))
        y_class: Workload classification labels (1D array of str)
        y_latency: True execution latencies in ms (1D array of float)
    """
    rng = np.random.default_rng(random_seed)
    samples_per_class = num_samples // 4

    X_list: List[np.ndarray] = []
    y_class_list: List[str] = []
    y_lat_list: List[float] = []

    # 1. GPU_INTENSIVE
    for _ in range(samples_per_class):
        batch_size = float(rng.choice([1, 2, 4, 8]))
        elements = float(rng.uniform(500000, 2000000)) * batch_size
        flops = float(rng.uniform(4.0, 15.0)) * batch_size
        mem = float(rng.uniform(35.0, 90.0)) * batch_size
        cpu = float(rng.uniform(15.0, 38.0))
        gpu = float(rng.uniform(78.0, 99.0))
        mem_util = float(rng.uniform(55.0, 92.0))
        io_wait = float(rng.uniform(1.0, 12.0))
        freq = float(rng.uniform(5.0, 25.0))
        latency = float(12.0 + (flops * 1.8) + (batch_size * 2.5) + rng.normal(0, 1.0))
        latency = max(5.0, latency)

        feats = [batch_size, elements, flops, mem, cpu, gpu, mem_util, io_wait, freq, latency]
        X_list.append(np.array(feats, dtype=np.float32))
        y_class_list.append(WorkloadClass.GPU_INTENSIVE.value)
        y_lat_list.append(latency)

    # 2. CPU_BOUND
    for _ in range(samples_per_class):
        batch_size = float(rng.choice([1, 2, 4, 8, 16]))
        elements = float(rng.uniform(100, 5000)) * batch_size
        flops = float(rng.uniform(0.001, 0.05)) * batch_size
        mem = float(rng.uniform(2.0, 8.0))
        cpu = float(rng.uniform(72.0, 98.0))
        gpu = float(rng.uniform(5.0, 22.0))
        mem_util = float(rng.uniform(15.0, 35.0))
        io_wait = float(rng.uniform(3.0, 18.0))
        freq = float(rng.uniform(20.0, 60.0))
        latency = float(0.8 + (batch_size * 0.15) + rng.normal(0, 0.2))
        latency = max(0.2, latency)

        feats = [batch_size, elements, flops, mem, cpu, gpu, mem_util, io_wait, freq, latency]
        X_list.append(np.array(feats, dtype=np.float32))
        y_class_list.append(WorkloadClass.CPU_BOUND.value)
        y_lat_list.append(latency)

    # 3. IO_BOUND
    for _ in range(samples_per_class):
        batch_size = float(rng.choice([1, 2]))
        elements = float(rng.uniform(1000, 32000)) * batch_size
        flops = float(rng.uniform(0.02, 0.2)) * batch_size
        mem = float(rng.uniform(8.0, 20.0))
        cpu = float(rng.uniform(12.0, 32.0))
        gpu = float(rng.uniform(8.0, 28.0))
        mem_util = float(rng.uniform(18.0, 40.0))
        io_wait = float(rng.uniform(52.0, 88.0))
        freq = float(rng.uniform(2.0, 15.0))
        latency = float(2.5 + (io_wait * 0.05) + rng.normal(0, 0.4))
        latency = max(1.0, latency)

        feats = [batch_size, elements, flops, mem, cpu, gpu, mem_util, io_wait, freq, latency]
        X_list.append(np.array(feats, dtype=np.float32))
        y_class_list.append(WorkloadClass.IO_BOUND.value)
        y_lat_list.append(latency)

    # 4. MIXED
    for _ in range(samples_per_class):
        batch_size = float(rng.choice([1, 2, 4]))
        elements = float(rng.uniform(50000, 200000)) * batch_size
        flops = float(rng.uniform(0.3, 2.0)) * batch_size
        mem = float(rng.uniform(15.0, 35.0)) * batch_size
        cpu = float(rng.uniform(38.0, 65.0))
        gpu = float(rng.uniform(42.0, 72.0))
        mem_util = float(rng.uniform(35.0, 65.0))
        io_wait = float(rng.uniform(10.0, 28.0))
        freq = float(rng.uniform(10.0, 35.0))
        latency = float(4.5 + (flops * 2.2) + rng.normal(0, 0.5))
        latency = max(2.0, latency)

        feats = [batch_size, elements, flops, mem, cpu, gpu, mem_util, io_wait, freq, latency]
        X_list.append(np.array(feats, dtype=np.float32))
        y_class_list.append(WorkloadClass.MIXED.value)
        y_lat_list.append(latency)

    # Shuffle synchronously
    indices = np.arange(len(X_list))
    rng.shuffle(indices)

    X = np.vstack(X_list)[indices]
    y_class = np.array(y_class_list)[indices]
    y_latency = np.array(y_lat_list)[indices]

    return X, y_class, y_latency


def save_dataset_to_csv(
    X: np.ndarray,
    y_class: np.ndarray,
    y_latency: np.ndarray,
    file_path: str,
) -> str:
    """Saves dataset matrix and labels to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    header = FEATURE_NAMES + ["workload_class", "target_latency_ms"]

    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row_x, c_lbl, lat in zip(X, y_class, y_latency):
            writer.writerow(list(row_x) + [c_lbl, float(lat)])

    return file_path


def load_dataset_from_csv(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads feature matrix and targets from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    X_rows: List[List[float]] = []
    y_class_rows: List[str] = []
    y_lat_rows: List[float] = []

    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            feats = [float(v) for v in row[:len(FEATURE_NAMES)]]
            c_lbl = str(row[len(FEATURE_NAMES)])
            lat = float(row[len(FEATURE_NAMES) + 1])
            X_rows.append(feats)
            y_class_rows.append(c_lbl)
            y_lat_rows.append(lat)

    return np.array(X_rows, dtype=np.float32), np.array(y_class_rows), np.array(y_lat_rows, dtype=np.float32)


def train_test_split_dataset(
    X: np.ndarray,
    y_class: np.ndarray,
    y_latency: np.ndarray,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits dataset into stratified train and test partitions to prevent data leakage."""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_class_train, y_class_test, y_lat_train, y_lat_test = train_test_split(
        X,
        y_class,
        y_latency,
        test_size=test_size,
        random_state=random_seed,
        stratify=y_class,
    )
    return X_train, X_test, y_class_train, y_class_test, y_lat_train, y_lat_test
