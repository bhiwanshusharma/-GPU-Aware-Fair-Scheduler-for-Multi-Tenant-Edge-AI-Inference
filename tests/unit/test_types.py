"""Unit tests for core enums and types."""

import pytest
from src.core.types import (
    WorkloadClass,
    JobState,
    PolicyType,
    SchedulingQuantum,
    PredictionOutput,
)


def test_workload_classes():
    assert WorkloadClass.GPU_INTENSIVE.value == "GPU_INTENSIVE"
    assert WorkloadClass.BALANCED.value == "BALANCED"
    assert WorkloadClass.CPU_BOUND.value == "CPU_BOUND"
    assert WorkloadClass.IO_BOUND.value == "IO_BOUND"


def test_policy_types():
    assert PolicyType.FIFO.value == "fifo"
    assert PolicyType.ROUND_ROBIN.value == "round_robin"
    assert PolicyType.STRICT_PRIORITY.value == "strict_priority"
    assert PolicyType.PDWFQ.value == "pdwfq"


def test_scheduling_quantum_validation():
    quantum = SchedulingQuantum(base_quantum_ms=25.0, min_quantum_ms=5.0, max_quantum_ms=50.0)
    assert quantum.base_quantum_ms == 25.0

    # Invalid: min > base
    with pytest.raises(ValueError):
        SchedulingQuantum(base_quantum_ms=10.0, min_quantum_ms=15.0)

    # Invalid: base > max
    with pytest.raises(ValueError):
        SchedulingQuantum(base_quantum_ms=60.0, max_quantum_ms=50.0)

    # Invalid: negative alpha
    with pytest.raises(ValueError):
        SchedulingQuantum(aging_alpha=-0.1)


def test_prediction_output_validation():
    pred = PredictionOutput(
        estimated_latency_ms=15.0,
        estimated_memory_mb=30.0,
        classification=WorkloadClass.GPU_INTENSIVE,
    )
    assert pred.estimated_latency_ms == 15.0

    with pytest.raises(ValueError):
        PredictionOutput(
            estimated_latency_ms=-1.0,
            estimated_memory_mb=10.0,
            classification=WorkloadClass.BALANCED,
        )
