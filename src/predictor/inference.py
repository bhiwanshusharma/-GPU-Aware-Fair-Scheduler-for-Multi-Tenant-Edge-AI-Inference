"""High-performance, low-overhead inference engine for runtime edge scheduling."""

import os
import logging
from typing import Optional, Dict, Any, List

from src.core.types import WorkloadClass, JobState, PredictionOutput
from src.core.exceptions import (
    ModelNotFoundError,
    PredictorUnavailableError,
    InvalidPredictorOutputError,
)
from src.core.job import GPUJob, Job
from src.core.tenant import Tenant
from src.predictor.base import BasePredictor
from src.predictor.features import WorkloadFeatureExtractor, JobFeatures
from src.predictor.model import LightweightWorkloadPredictor

logger = logging.getLogger(__name__)


class WorkloadInferenceEngine(BasePredictor):
    """Provides high-throughput, low-latency prediction for runtime GPU scheduling.
    
    Includes graceful heuristic fallback when ML models are unavailable or uninitialized.
    """

    def __init__(
        self,
        predictor: Optional[LightweightWorkloadPredictor] = None,
        feature_extractor: Optional[WorkloadFeatureExtractor] = None,
        enabled: bool = True,
    ) -> None:
        self.predictor = predictor
        self.feature_extractor = feature_extractor or WorkloadFeatureExtractor()
        self.enabled = enabled
        self.total_inferences = 0

    def predict(self, job: Job) -> PredictionOutput:
        """Implements BasePredictor abstract method."""
        return self.predict_job(job)

    def calibrate(self, calibration_traces: list[dict]) -> None:
        """Calibrates model using recorded traces if supported."""
        pass

    @classmethod
    def from_model_file(
        cls,
        model_path: str,
        enabled: bool = True,
        strict: bool = False,
    ) -> "WorkloadInferenceEngine":
        """Instantiates an inference engine loaded from a saved model artifact.
        
        Args:
            model_path: Path to serialized model artifact.
            enabled: Whether predictor should be active.
            strict: If True, raises ModelNotFoundError on missing/corrupted file.
        """
        if os.path.exists(model_path):
            try:
                predictor = LightweightWorkloadPredictor.load(model_path)
                logger.info(f"Loaded trained workload predictor from {model_path}")
                return cls(predictor=predictor, enabled=enabled)
            except Exception as e:
                if strict:
                    raise ModelNotFoundError(f"Failed to load model from {model_path}: {e}")
                logger.warning(f"Failed to load model from {model_path} ({e}); falling back to heuristic engine.")
        else:
            if strict:
                raise ModelNotFoundError(f"Model artifact {model_path} does not exist.")
            logger.warning(f"Model artifact {model_path} does not exist; running in fallback mode.")
        return cls(predictor=None, enabled=enabled)

    def predict_job(
        self,
        job: GPUJob,
        tenant: Optional[Tenant] = None,
        system_context: Optional[Dict[str, float]] = None,
    ) -> PredictionOutput:
        """Predicts workload characteristics and annotates job metadata.
        
        Args:
            job: Target inference job.
            tenant: Optional tenant context.
            system_context: Optional real-time system context metrics.
            
        Returns:
            PredictionOutput containing predicted classification and estimated latency.
        """
        self.total_inferences += 1

        # Fallback heuristic if ML model is disabled or not loaded
        if not self.enabled or self.predictor is None or not self.predictor.is_trained:
            return self._heuristic_fallback(job)

        try:
            features = self.feature_extractor.extract(job, tenant, system_context)
            prediction = self.predictor.predict_single(features)

            # Annotate job with predicted attributes
            job.workload_type = prediction.classification
            job.predicted_memory_mb = prediction.estimated_memory_mb
            job.metadata["predicted_latency_ms"] = prediction.estimated_latency_ms
            job.metadata["prediction_confidence"] = prediction.confidence_score
            if job.status == JobState.QUEUED:
                job.status = JobState.PREDICTED

            return prediction
        except Exception as e:
            logger.warning(f"Predictor exception during inference: {e}. Using fallback.")
            return self._heuristic_fallback(job)

    def predict_strict(
        self,
        job: GPUJob,
        tenant: Optional[Tenant] = None,
        system_context: Optional[Dict[str, float]] = None,
    ) -> PredictionOutput:
        """Runs strict prediction, raising PredictorUnavailableError or InvalidPredictorOutputError on failures."""
        if not self.enabled or self.predictor is None or not self.predictor.is_trained:
            raise PredictorUnavailableError(
                "Predictor is unavailable, disabled, or uninitialized; strict inference failed."
            )
        features = self.feature_extractor.extract(job, tenant, system_context)
        return self.predictor.predict_single(features)

    def _heuristic_fallback(self, job: GPUJob) -> PredictionOutput:
        """Rule-based heuristic fallback guaranteeing zero scheduler disruption."""
        model_name = job.model_name.lower()
        if "yolo" in model_name or "detect" in model_name or "resnet" in model_name:
            wl_class = WorkloadClass.GPU_INTENSIVE
            est_lat = float(job.estimated_gpu_time or 22.5)
            est_mem = 45.0
        elif "sensor" in model_name or "mlp" in model_name:
            wl_class = WorkloadClass.CPU_BOUND
            est_lat = float(job.estimated_gpu_time or 1.2)
            est_mem = 4.0
        elif "audio" in model_name or "stream" in model_name or "kws" in model_name:
            wl_class = WorkloadClass.IO_BOUND
            est_lat = float(job.estimated_gpu_time or 3.0)
            est_mem = 12.0
        else:
            wl_class = WorkloadClass.MIXED
            est_lat = float(job.estimated_gpu_time or 5.5)
            est_mem = 18.0

        return PredictionOutput(
            estimated_latency_ms=est_lat,
            estimated_memory_mb=est_mem,
            classification=wl_class,
            confidence_score=0.75,
        )
