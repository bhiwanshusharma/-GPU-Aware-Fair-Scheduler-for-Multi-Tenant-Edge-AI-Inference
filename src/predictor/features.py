"""Feature extraction for the AI Workload Predictor."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np

from src.core.job import GPUJob
from src.core.tenant import Tenant


FEATURE_NAMES = [
    "batch_size",
    "input_elements",
    "model_complexity_flops_g",
    "memory_footprint_mb",
    "cpu_utilization_pct",
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "io_wait_pct",
    "request_frequency_hz",
    "hist_avg_latency_ms",
]


@dataclass
class JobFeatures:
    """Normalized feature vector extracted from an inference job and system state."""
    batch_size: float
    input_elements: float
    model_complexity_flops_g: float
    memory_footprint_mb: float
    cpu_utilization_pct: float
    gpu_utilization_pct: float
    memory_utilization_pct: float
    io_wait_pct: float
    request_frequency_hz: float
    hist_avg_latency_ms: float

    def to_array(self) -> np.ndarray:
        """Converts feature container to a 1D numpy array."""
        return np.array([
            self.batch_size,
            self.input_elements,
            self.model_complexity_flops_g,
            self.memory_footprint_mb,
            self.cpu_utilization_pct,
            self.gpu_utilization_pct,
            self.memory_utilization_pct,
            self.io_wait_pct,
            self.request_frequency_hz,
            self.hist_avg_latency_ms,
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        """Converts feature container to a dictionary."""
        return {name: float(val) for name, val in zip(FEATURE_NAMES, self.to_array())}


class WorkloadFeatureExtractor:
    """Extracts operational feature vectors from GPU jobs, tenant history, and system context."""

    # Default knowledge base for well-known edge AI models
    MODEL_PROFILES = {
        "yolov8n": {"flops_g": 8.7, "memory_mb": 45.0, "elements": 3 * 640 * 640},
        "yolov8n_detection": {"flops_g": 8.7, "memory_mb": 45.0, "elements": 3 * 640 * 640},
        "mobilenetv3": {"flops_g": 0.44, "memory_mb": 18.0, "elements": 3 * 224 * 224},
        "mobilenetv3_classification": {"flops_g": 0.44, "memory_mb": 18.0, "elements": 3 * 224 * 224},
        "mobilenet": {"flops_g": 0.44, "memory_mb": 18.0, "elements": 3 * 224 * 224},
        "keyword_spotting": {"flops_g": 0.08, "memory_mb": 12.0, "elements": 16000},
        "keyword_spotting_audio": {"flops_g": 0.08, "memory_mb": 12.0, "elements": 16000},
        "audio": {"flops_g": 0.08, "memory_mb": 12.0, "elements": 16000},
        "sensor_anomaly": {"flops_g": 0.002, "memory_mb": 4.0, "elements": 128},
        "sensor_anomaly_mlp": {"flops_g": 0.002, "memory_mb": 4.0, "elements": 128},
    }

    def __init__(self) -> None:
        pass

    def extract(
        self,
        job: GPUJob,
        tenant: Optional[Tenant] = None,
        system_context: Optional[Dict[str, float]] = None,
    ) -> JobFeatures:
        """Extracts features for an individual inference job.
        
        Args:
            job: The GPUJob instance.
            tenant: Optional Tenant instance providing historical context.
            system_context: Optional real-time system metrics (CPU, GPU, memory, IO).
            
        Returns:
            JobFeatures instance ready for model inference.
        """
        ctx = system_context or {}
        model_name = job.model_name.lower().strip()
        default_profile = self.MODEL_PROFILES.get(
            model_name,
            {"flops_g": 1.0, "memory_mb": 20.0, "elements": 150528}
        )

        # 1. Batch size
        batch_size = float(max(1, job.batch_size))

        # 2. Input elements
        if job.input_shape:
            input_elements = float(math.prod(job.input_shape))
        else:
            input_elements = float(default_profile["elements"]) * batch_size

        # 3. Model complexity FLOPs
        flops_g = float(job.metadata.get("flops_g", default_profile["flops_g"])) * batch_size

        # 4. Memory footprint
        memory_mb = float(
            job.predicted_memory_mb
            or job.metadata.get("memory_mb", default_profile["memory_mb"])
        )

        # 5. CPU utilization (derived from context or job metadata)
        cpu_util = float(ctx.get("cpu_utilization_pct", job.metadata.get("cpu_utilization_pct", 30.0)))

        # 6. GPU utilization
        gpu_util = float(ctx.get("gpu_utilization_pct", job.metadata.get("gpu_utilization_pct", 50.0)))

        # 7. Memory bandwidth utilization
        mem_util = float(ctx.get("memory_utilization_pct", job.metadata.get("memory_utilization_pct", 35.0)))

        # 8. I/O wait
        io_wait = float(ctx.get("io_wait_pct", job.metadata.get("io_wait_pct", 10.0)))

        # 9. Request frequency
        req_freq = float(ctx.get("request_frequency_hz", job.metadata.get("request_frequency_hz", 15.0)))

        # 10. Historical average latency
        if tenant and tenant.total_jobs_completed > 0:
            hist_lat = tenant.accumulated_gpu_service / max(1, tenant.total_jobs_completed)
        else:
            hist_lat = float(job.estimated_gpu_time or 15.0)

        return JobFeatures(
            batch_size=batch_size,
            input_elements=input_elements,
            model_complexity_flops_g=flops_g,
            memory_footprint_mb=memory_mb,
            cpu_utilization_pct=cpu_util,
            gpu_utilization_pct=gpu_util,
            memory_utilization_pct=mem_util,
            io_wait_pct=io_wait,
            request_frequency_hz=req_freq,
            hist_avg_latency_ms=hist_lat,
        )
