"""Training engine — pipeline parallelism, gradient aggregation, and orchestration."""

from imo.training.aggregator import (
    AggregationStrategy,
    FederatedAveraging,
    Krum,
    TrimmedMean,
)
from imo.training.checkpoint import CheckpointManager, CheckpointMetadata
from imo.training.engine import (
    DistributedTrainingEngine,
    TrainingConfig,
    TrainingMetrics,
    TrainingStatus,
)
from imo.training.pipeline import (
    MicrobatchScheduler,
    PipelineParallelism,
    PipelineStage,
    create_pipeline_model,
)
from imo.training.security import (
    ByzantineRobustAggregator,
    PoisoningDetector,
    SecurityAlert,
)
from imo.training.verifier import GradientAnomalyDetector, GradientVerifier, VerificationResult

__all__ = [
    "DistributedTrainingEngine",
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingStatus",
    "PipelineParallelism",
    "PipelineStage",
    "MicrobatchScheduler",
    "create_pipeline_model",
    "AggregationStrategy",
    "FederatedAveraging",
    "TrimmedMean",
    "Krum",
    "CheckpointManager",
    "CheckpointMetadata",
    "GradientVerifier",
    "VerificationResult",
    "GradientAnomalyDetector",
    "PoisoningDetector",
    "SecurityAlert",
    "ByzantineRobustAggregator",
]
