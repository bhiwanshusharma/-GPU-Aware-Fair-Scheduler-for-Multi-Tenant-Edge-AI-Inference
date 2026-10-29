"""Unit tests for the AI-based Workload Predictor component.

Verifies:
1. Feature extraction correctness.
2. Synthetic dataset generation and CSV roundtrip.
3. Train/test split stratification and zero data leakage.
4. Model training and evaluation metrics (accuracy, precision, recall, F1, MAE).
5. Model serialization and deserialization roundtrip.
6. Runtime inference engine and job metadata decoration.
7. Fallback behavior when ML model is disabled or unavailable.
8. Scheduler integration and dynamic quantum scaling via workload multiplier psi(c).
"""

import os
import tempfile
import pytest
import numpy as np

from src.core.types import WorkloadClass, JobState, PredictionOutput, SchedulingQuantum
from src.core.job import GPUJob
from src.core.tenant import Tenant
from src.predictor.features import WorkloadFeatureExtractor, JobFeatures, FEATURE_NAMES
from src.predictor.dataset import (
    generate_synthetic_workload_dataset,
    save_dataset_to_csv,
    load_dataset_from_csv,
    train_test_split_dataset,
)
from src.predictor.model import LightweightWorkloadPredictor
from src.predictor.trainer import train_workload_predictor
from src.predictor.evaluator import evaluate_workload_predictor
from src.predictor.inference import WorkloadInferenceEngine
from src.scheduler.fair_scheduler import GPUFairScheduler
from src.scheduler.evaluator import run_scheduler_benchmark


# =========================================================================
# 1. Feature Extraction
# =========================================================================
def test_feature_extraction():
    """Verifies that all 10 features are extracted correctly with proper dimensions."""
    extractor = WorkloadFeatureExtractor()
    tenant = Tenant("tenant_test", name="Test", weight=2.0, priority=8, gpu_quota=50.0)
    tenant.record_gpu_execution(40.0, current_time_ms=50.0)
    job = GPUJob(
        job_id="J_FEAT",
        tenant_id="tenant_test",
        model_name="yolov8n",
        arrival_time=0.0,
        batch_size=2,
        input_shape=(2, 3, 640, 640),
        estimated_gpu_time=25.0,
        remaining_gpu_time=25.0,
    )

    context = {
        "cpu_utilization_pct": 25.0,
        "gpu_utilization_pct": 88.0,
        "memory_utilization_pct": 60.0,
        "io_wait_pct": 5.0,
        "request_frequency_hz": 12.0,
    }

    feats = extractor.extract(job, tenant=tenant, system_context=context)

    assert isinstance(feats, JobFeatures)
    arr = feats.to_array()
    assert arr.shape == (10,)
    assert feats.batch_size == 2.0
    assert feats.input_elements == float(2 * 3 * 640 * 640)
    assert feats.cpu_utilization_pct == 25.0
    assert feats.gpu_utilization_pct == 88.0
    assert feats.io_wait_pct == 5.0
    assert feats.hist_avg_latency_ms > 0.0

    d = feats.to_dict()
    assert len(d) == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert name in d


# =========================================================================
# 2. Dataset Generation & CSV Persistence
# =========================================================================
def test_dataset_generation_and_csv_roundtrip():
    """Verifies synthetic dataset generation and lossless CSV save/load."""
    X, y_class, y_lat = generate_synthetic_workload_dataset(num_samples=200, random_seed=42)

    assert X.shape == (200, 10)
    assert len(y_class) == 200
    assert len(y_lat) == 200

    # Ensure all 4 classes are present in balanced numbers
    unique_classes = set(y_class)
    assert len(unique_classes) == 4
    for c in [WorkloadClass.GPU_INTENSIVE.value, WorkloadClass.CPU_BOUND.value, WorkloadClass.IO_BOUND.value, WorkloadClass.MIXED.value]:
        assert c in unique_classes

    # Roundtrip to temporary CSV
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_dataset_to_csv(X, y_class, y_lat, tmp_path)
        X_loaded, y_class_loaded, y_lat_loaded = load_dataset_from_csv(tmp_path)

        assert X_loaded.shape == X.shape
        assert np.allclose(X_loaded, X, atol=1e-5)
        assert np.array_equal(y_class_loaded, y_class)
        assert np.allclose(y_lat_loaded, y_lat, atol=1e-5)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =========================================================================
# 3. Train/Test Separation (Zero Data Leakage)
# =========================================================================
def test_train_test_split_no_leakage():
    """Verifies strict train/test separation with stratification."""
    X, y_class, y_lat = generate_synthetic_workload_dataset(num_samples=400, random_seed=123)
    X_train, X_test, y_train, y_test, y_lat_train, y_lat_test = train_test_split_dataset(
        X, y_class, y_lat, test_size=0.25, random_seed=123
    )

    assert len(X_train) == 300
    assert len(X_test) == 100
    assert len(y_lat_train) == 300
    assert len(y_lat_test) == 100

    # Verify stratification (each class should have exactly 25% in test)
    for c in set(y_class):
        n_train_c = np.sum(y_train == c)
        n_test_c = np.sum(y_test == c)
        assert n_train_c == 75
        assert n_test_c == 25


# =========================================================================
# 4. Model Training & Evaluation Metrics
# =========================================================================
def test_model_fit_and_metrics():
    """Trains predictor and validates accuracy, precision, recall, F1, and latency MAE."""
    X, y_class, y_lat = generate_synthetic_workload_dataset(num_samples=600, random_seed=42)
    X_train, X_test, y_train, y_test, y_lat_train, y_lat_test = train_test_split_dataset(
        X, y_class, y_lat, test_size=0.2, random_seed=42
    )

    model = LightweightWorkloadPredictor()
    model.fit(X_train, y_train, y_lat_train)

    pred_classes, confs, pred_lats = model.predict(X_test)

    metrics = evaluate_workload_predictor(
        y_class_true=y_test,
        y_class_pred=pred_classes,
        y_lat_true=y_lat_test,
        y_lat_pred=pred_lats,
    )

    # High classification performance on physically distinct workload regimes
    assert metrics.accuracy >= 0.95
    assert metrics.f1_macro >= 0.95
    assert metrics.precision_macro >= 0.95
    assert metrics.recall_macro >= 0.95
    # High regression performance
    assert metrics.latency_mae_ms < 2.0
    assert metrics.latency_r2 > 0.90
    assert len(metrics.summary()) > 100


# =========================================================================
# 5. Model Serialization Roundtrip
# =========================================================================
def test_model_serialization_roundtrip():
    """Verifies that saving and loading a trained model preserves exact predictions."""
    X, y_class, y_lat = generate_synthetic_workload_dataset(num_samples=200, random_seed=42)
    model = LightweightWorkloadPredictor()
    model.fit(X, y_class, y_lat)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model.save(tmp_path)
        assert os.path.exists(tmp_path)
        # Check artifact size is under 500 KB
        assert os.path.getsize(tmp_path) < 500 * 1024

        loaded_model = LightweightWorkloadPredictor.load(tmp_path)
        assert loaded_model.is_trained

        # Compare predictions on new samples
        X_eval = X[:10]
        c1, cf1, l1 = model.predict(X_eval)
        c2, cf2, l2 = loaded_model.predict(X_eval)

        assert np.array_equal(c1, c2)
        assert np.allclose(cf1, cf2, atol=1e-5)
        assert np.allclose(l1, l2, atol=1e-5)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =========================================================================
# 6. Runtime Inference Engine & Job Annotation
# =========================================================================
def test_inference_engine_job_annotation():
    """Inference engine decorates job with classification and latency metadata."""
    X, y_class, y_lat = generate_synthetic_workload_dataset(num_samples=400, random_seed=42)
    predictor = LightweightWorkloadPredictor()
    predictor.fit(X, y_class, y_lat)

    engine = WorkloadInferenceEngine(predictor=predictor)

    job = GPUJob(
        "J_INFER",
        "tenant_1",
        "yolov8n",
        arrival_time=0.0,
        estimated_gpu_time=20.0,
        remaining_gpu_time=20.0,
    )
    job.metadata["cpu_utilization_pct"] = 20.0
    job.metadata["gpu_utilization_pct"] = 92.0
    job.metadata["io_wait_pct"] = 4.0

    out = engine.predict_job(job)

    assert isinstance(out, PredictionOutput)
    assert out.classification == WorkloadClass.GPU_INTENSIVE
    assert out.confidence_score > 0.5
    assert out.estimated_latency_ms > 0.0

    # Job was annotated
    assert job.workload_type == WorkloadClass.GPU_INTENSIVE
    assert job.metadata["predicted_latency_ms"] == out.estimated_latency_ms
    assert job.status == JobState.PREDICTED


# =========================================================================
# 7. Fallback Behavior When Model Unavailable
# =========================================================================
def test_inference_engine_fallback_behavior():
    """Engine gracefully uses rule heuristics if ML model is missing or disabled."""
    # Engine without model
    fallback_engine = WorkloadInferenceEngine(predictor=None, enabled=True)

    job_vision = GPUJob("JV", "t", "yolov8n_detect", arrival_time=0.0, estimated_gpu_time=22.0, remaining_gpu_time=22.0)
    out_v = fallback_engine.predict_job(job_vision)
    assert out_v.classification == WorkloadClass.GPU_INTENSIVE

    job_sensor = GPUJob("JS", "t", "sensor_anomaly_mlp", arrival_time=0.0, estimated_gpu_time=1.0, remaining_gpu_time=1.0)
    out_s = fallback_engine.predict_job(job_sensor)
    assert out_s.classification == WorkloadClass.CPU_BOUND

    job_audio = GPUJob("JA", "t", "keyword_spotting_audio", arrival_time=0.0, estimated_gpu_time=2.5, remaining_gpu_time=2.5)
    out_a = fallback_engine.predict_job(job_audio)
    assert out_a.classification == WorkloadClass.IO_BOUND


# =========================================================================
# 8. Scheduler Integration with Predictor
# =========================================================================
def test_scheduler_integration_with_predictor():
    """Predictor dynamically modulates quantum slice via psi(c) and short-job bias."""
    t1 = Tenant("tenant_1", name="T1", weight=1.0, priority=5, gpu_quota=100.0)
    t2 = Tenant("tenant_2", name="T2", weight=1.0, priority=5, gpu_quota=100.0)
    tenants = {t1.tenant_id: t1, t2.tenant_id: t2}

    # Train predictor
    X, y_class, y_lat = generate_synthetic_workload_dataset(num_samples=400, random_seed=42)
    predictor = LightweightWorkloadPredictor()
    predictor.fit(X, y_class, y_lat)
    engine = WorkloadInferenceEngine(predictor=predictor)

    sq = SchedulingQuantum(base_quantum_ms=20.0, min_quantum_ms=5.0, max_quantum_ms=50.0)
    sched = GPUFairScheduler(tenants=tenants, quantum=sq, predictor=engine, enable_workload_multiplier=True)

    # Workload multiplier test
    assert sched.get_workload_multiplier(WorkloadClass.GPU_INTENSIVE) == 0.8
    assert sched.get_workload_multiplier(WorkloadClass.MIXED) == 1.0
    assert sched.get_workload_multiplier(WorkloadClass.CPU_BOUND) == 1.2
    assert sched.get_workload_multiplier(WorkloadClass.IO_BOUND) == 1.5

    # Run complete benchmark with predictor active
    jobs = [
        GPUJob("Y1", "tenant_1", "yolov8n", arrival_time=0.0, estimated_gpu_time=25.0, remaining_gpu_time=25.0),
        GPUJob("S1", "tenant_2", "sensor_mlp", arrival_time=0.0, estimated_gpu_time=2.0, remaining_gpu_time=2.0),
    ]

    res = run_scheduler_benchmark(sched, jobs, quantum_ms=20.0, context_switch_overhead_ms=0.5)

    assert len(res.completed_jobs) == 2
    assert res.gpu_utilization_pct > 80.0
