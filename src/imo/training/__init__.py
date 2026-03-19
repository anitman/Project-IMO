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
    ActivationCache,
    BlockInfo,
    BlockRebalancer,
    BlockServer,
    MicrobatchScheduler,
    PipelineRouter,
    PipelineStage,
    RemoteBlock,
    RemoteSequential,
    ServerStatus,
    build_remote_pipeline,
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
    "PipelineStage",
    "MicrobatchScheduler",
    "create_pipeline_model",
    "BlockServer",
    "RemoteBlock",
    "RemoteSequential",
    "PipelineRouter",
    "BlockRebalancer",
    "BlockInfo",
    "ServerStatus",
    "ActivationCache",
    "build_remote_pipeline",
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
