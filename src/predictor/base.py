"""Abstract base class for AI Workload Prediction."""

from abc import ABC, abstractmethod
from src.core.types import PredictionOutput, WorkloadClass
from src.core.job import Job


class BasePredictor(ABC):
    """Abstract interface for workload characteristics and execution time predictor."""

    @abstractmethod
    def predict(self, job: Job) -> PredictionOutput:
        """Predicts GPU execution time, memory footprint, and workload category for a job.
        
        Args:
            job: Candidate Job instance before dispatch.
            
        Returns:
            PredictionOutput containing estimated latency, memory, and classification.
        """
        pass

    @abstractmethod
    def calibrate(self, calibration_traces: list[dict]) -> None:
        """Calibrates or trains the prediction model using recorded execution traces."""
        pass
