"""Contribution scoring and reward calculation.

Key principles:
1. No upfront staking required - rewards are quality-based after training
2. Rewards distributed after model evaluation by community
3. Quality score determines total reward pool eligibility
4. Encourages high-quality open-source models, not reward farming
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

ContributionScores: TypeAlias = dict[str, float]


class ModelQualityLevel(Enum):
    """Model quality assessment levels."""

    POOR = "poor"  # Below acceptable threshold
    FAIR = "fair"  # Meets minimum standards
    GOOD = "good"  # Competitive performance
    EXCELLENT = "excellent"  # SOTA-level performance
    BREAKTHROUGH = "breakthrough"  # Surpasses SOTA


@dataclass
class ModelEvaluation:
    """Model quality evaluation results."""

    benchmark_scores: dict[str, float]  # e.g., {"MMLU": 68.5, "GSM8K": 72.3}
    community_rating: float  # 0.0-1.0 from community voting
    sota_comparison: float  # 0.0-1.0, where 1.0 = matches SOTA
    code_quality: float = 0.8  # 0.0-1.0, reproducibility assessment
    documentation_quality: float = 0.8  # 0.0-1.0

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score (0.0-1.0)."""
        # Weighted average of components
        benchmark_avg = (
            sum(self.benchmark_scores.values()) / len(self.benchmark_scores)
            if self.benchmark_scores
            else 0
        ) / 100  # Assume benchmarks are 0-100

        return (
            benchmark_avg * 0.40  # 40% benchmark performance
            + self.community_rating * 0.25  # 25% community approval
            + self.sota_comparison * 0.25  # 25% vs SOTA
            + self.code_quality * 0.05  # 5% code quality
            + self.documentation_quality * 0.05  # 5% documentation
        )

    @property
    def quality_level(self) -> ModelQualityLevel:
        """Determine quality level based on overall score."""
        score = self.overall_score

        if score >= 0.95:
            return ModelQualityLevel.BREAKTHROUGH
        elif score >= 0.85:
            return ModelQualityLevel.EXCELLENT
        elif score >= 0.70:
            return ModelQualityLevel.GOOD
        elif score >= 0.50:
            return ModelQualityLevel.FAIR
        else:
            return ModelQualityLevel.POOR


@dataclass
class Contribution:
    """Contribution record for a node."""

    node_id: str
    compute_time: float
    loss_reduction: float
    vram_gb: int
    verified: bool = True
    reputation: float = 1.0


@dataclass
class RewardPool:
    """Reward pool for a completed IMO."""

    imo_id: int
    model_category: str
    total_pool: float  # Base pool before quality adjustment
    quality_multiplier: float  # Based on model evaluation
    adjusted_pool: float  # total_pool * quality_multiplier
    data_pool: float  # 40%
    compute_pool: float  # 50%
    paper_pool: float  # 10%
    is_distributed: bool = False


class ContributionCalculator:
    """Calculate contribution scores and rewards.

    Reward Flow:
    1. Training completes → model submitted for evaluation
    2. Community evaluates model quality (benchmarks + voting)
    3. Quality score determines reward multiplier (0.0-3.0x)
    4. Adjusted pool distributed to contributors
    5. Low quality models (<0.5) receive minimal or no rewards
    """

    # Quality multipliers
    QUALITY_MULTIPLIERS = {
        ModelQualityLevel.POOR: 0.0,  # No reward - below threshold
        ModelQualityLevel.FAIR: 0.5,  # 50% of base pool
        ModelQualityLevel.GOOD: 1.0,  # 100% of base pool
        ModelQualityLevel.EXCELLENT: 1.5,  # 150% of base pool
        ModelQualityLevel.BREAKTHROUGH: 2.0,  # 200% of base pool
    }

    # Minimum threshold for any rewards
    MIN_QUALITY_THRESHOLD = 0.50

    def __init__(self, base_reward_pool: float):
        self.base_reward_pool = base_reward_pool

    def evaluate_model(
        self,
        benchmark_scores: dict[str, float],
        community_rating: float,
        sota_comparison: float,
        code_quality: float = 0.8,
        documentation_quality: float = 0.8,
    ) -> ModelEvaluation:
        """Evaluate model quality and determine reward eligibility."""
        evaluation = ModelEvaluation(
            benchmark_scores=benchmark_scores,
            community_rating=community_rating,
            sota_comparison=sota_comparison,
            code_quality=code_quality,
            documentation_quality=documentation_quality,
        )

        return evaluation

    def calculate_reward_pool(
        self,
        evaluation: ModelEvaluation,
    ) -> RewardPool:
        """Calculate adjusted reward pool based on model quality."""
        quality_level = evaluation.quality_level
        multiplier = self.QUALITY_MULTIPLIERS.get(quality_level, 0.0)

        # If below minimum threshold, no rewards
        if evaluation.overall_score < self.MIN_QUALITY_THRESHOLD:
            multiplier = 0.0

        adjusted_pool = self.base_reward_pool * multiplier

        return RewardPool(
            imo_id=0,  # Set by caller
            model_category="",  # Set by caller
            total_pool=self.base_reward_pool,
            quality_multiplier=multiplier,
            adjusted_pool=adjusted_pool,
            data_pool=adjusted_pool * 0.40,
            compute_pool=adjusted_pool * 0.50,
            paper_pool=adjusted_pool * 0.10,
            is_distributed=False,
        )

    def compute_contribution_score(
        self,
        compute_time: float,
        loss_reduction: float,
        vram_gb: int,
        total_vram: int,
    ) -> float:
        """Compute contribution score for a node.

        C_i = (T_i × ΔL_eff_i) × (M_vram_i / M_total)
        """
        time_component = compute_time
        loss_component = max(0, loss_reduction)
        vram_ratio = vram_gb / total_vram if total_vram > 0 else 0

        return time_component * loss_component * vram_ratio

    def distribute_compute_rewards(
        self,
        contributions: list[Contribution],
        compute_pool: float,
    ) -> dict[str, float]:
        """Distribute compute rewards among contributors."""
        total_vram = sum(c.vram_gb for c in contributions)

        scores: dict[str, float] = {}
        for contrib in contributions:
            if not contrib.verified:
                scores[contrib.node_id] = 0.0
                continue

            score = self.compute_contribution_score(
                compute_time=contrib.compute_time,
                loss_reduction=contrib.loss_reduction,
                vram_gb=contrib.vram_gb,
                total_vram=total_vram,
            )
            scores[contrib.node_id] = score * contrib.reputation

        total_score = sum(scores.values())

        if total_score == 0:
            return {node_id: 0.0 for node_id in scores}

        rewards: dict[str, float] = {}
        for node_id, score in scores.items():
            share = score / total_score
            rewards[node_id] = share * compute_pool

        return rewards

    def distribute_data_rewards(
        self,
        dataset_contributions: list[dict],
        data_pool: float,
    ) -> dict[str, float]:
        """Distribute data rewards among dataset contributors.

        Based on:
        - Number of samples contributed
        - Quality score of dataset
        - Uniqueness (deduplication factor)
        """
        total_weighted_samples = 0
        weights: dict[str, float] = {}

        for contrib in dataset_contributions:
            samples = contrib.get("num_samples", 0)
            quality = contrib.get("quality_score", 0.5)
            uniqueness = contrib.get("uniqueness_factor", 1.0)

            weighted = samples * quality * uniqueness
            weights[contrib["dataset_id"]] = weighted
            total_weighted_samples += weighted

        if total_weighted_samples == 0:
            return {cid: 0.0 for cid in weights}

        rewards: dict[str, float] = {}
        for dataset_id, weight in weights.items():
            share = weight / total_weighted_samples
            rewards[dataset_id] = share * data_pool

        return rewards

    def distribute_paper_rewards(
        self,
        authors: list[str],
        paper_pool: float,
        author_contributions: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Distribute paper rewards among authors.

        If author_contributions provided, split proportionally.
        Otherwise, split equally.
        """
        if not authors:
            return {}

        if author_contributions and len(author_contributions) == len(authors):
            total = sum(author_contributions.values())
            if total == 0:
                return {a: paper_pool / len(authors) for a in authors}
            return {
                author: (author_contributions[author] / total) * paper_pool for author in authors
            }
        else:
            # Equal split
            share = paper_pool / len(authors)
            return {author: share for author in authors}


    def distribute_all_rewards(
        self,
        evaluation: ModelEvaluation,
        compute_contributions: list[Contribution],
        dataset_contributions: list[dict],
        paper_authors: list[str],
        author_contributions: dict[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Distribute all rewards (compute + data + paper) in one call.

        Returns a dict keyed by pool type ("compute", "data", "paper"),
        each mapping contributor ID → reward amount.
        """
        pool = self.calculate_reward_pool(evaluation)

        return {
            "compute": self.distribute_compute_rewards(
                compute_contributions, pool.compute_pool
            ),
            "data": self.distribute_data_rewards(
                dataset_contributions, pool.data_pool
            ),
            "paper": self.distribute_paper_rewards(
                paper_authors, pool.paper_pool, author_contributions
            ),
            "_pool": {
                "total": pool.total_pool,
                "adjusted": pool.adjusted_pool,
                "multiplier": pool.quality_multiplier,
                "quality_level": evaluation.quality_level.value,
            },
        }
