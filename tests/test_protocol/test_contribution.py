"""Tests for contribution scoring and reward calculation."""

from __future__ import annotations

import pytest

from imo.protocol.contribution import (
    Contribution,
    ContributionCalculator,
    ModelQualityLevel,
    ModelEvaluation,
)


class TestModelEvaluation:
    """Test ModelQualityLevel and ModelEvaluation."""

    def test_breakthrough_quality(self) -> None:
        """Test breakthrough model evaluation."""
        eval_result = ModelEvaluation(
            benchmark_scores={"MMLU": 85.2, "GSM8K": 92.1},
            community_rating=0.98,
            sota_comparison=1.10,
            code_quality=0.95,
            documentation_quality=0.92,
        )

        assert eval_result.quality_level == ModelQualityLevel.BREAKTHROUGH
        assert eval_result.overall_score >= 0.90

    def test_excellent_quality(self) -> None:
        """Test excellent model evaluation."""
        eval_result = ModelEvaluation(
            benchmark_scores={"MMLU": 78.5, "GSM8K": 85.3},
            community_rating=0.90,
            sota_comparison=1.00,
            code_quality=0.88,
            documentation_quality=0.85,
        )

        assert eval_result.quality_level == ModelQualityLevel.EXCELLENT

    def test_good_quality(self) -> None:
        """Test good model evaluation."""
        eval_result = ModelEvaluation(
            benchmark_scores={"MMLU": 68.2, "GSM8K": 72.5},
            community_rating=0.75,
            sota_comparison=0.88,
            code_quality=0.80,
            documentation_quality=0.78,
        )

        assert eval_result.quality_level == ModelQualityLevel.GOOD

    def test_fair_quality(self) -> None:
        """Test fair model evaluation."""
        eval_result = ModelEvaluation(
            benchmark_scores={"MMLU": 55.3, "GSM8K": 48.2},
            community_rating=0.55,
            sota_comparison=0.70,
            code_quality=0.70,
            documentation_quality=0.65,
        )

        assert eval_result.quality_level == ModelQualityLevel.FAIR

    def test_poor_quality(self) -> None:
        """Test poor model evaluation."""
        eval_result = ModelEvaluation(
            benchmark_scores={"MMLU": 35.2, "GSM8K": 28.5},
            community_rating=0.30,
            sota_comparison=0.45,
            code_quality=0.50,
            documentation_quality=0.40,
        )

        assert eval_result.quality_level == ModelQualityLevel.POOR


class TestContributionCalculator:
    """Test ContributionCalculator functionality."""

    def test_quality_multiplier_breakthrough(self) -> None:
        """Test breakthrough model gets 2x multiplier."""
        calculator = ContributionCalculator(base_reward_pool=1_000_000)

        evaluation = ModelEvaluation(
            benchmark_scores={"MMLU": 85.2, "GSM8K": 92.1},
            community_rating=0.98,
            sota_comparison=1.10,
        )

        pool = calculator.calculate_reward_pool(evaluation)
        assert pool.quality_multiplier == 2.0
        assert pool.adjusted_pool == 2_000_000

    def test_quality_multiplier_excellent(self) -> None:
        """Test excellent model gets 1.5x multiplier."""
        calculator = ContributionCalculator(base_reward_pool=1_000_000)

        evaluation = ModelEvaluation(
            benchmark_scores={"MMLU": 78.5, "GSM8K": 85.3},
            community_rating=0.90,
            sota_comparison=1.00,
        )

        pool = calculator.calculate_reward_pool(evaluation)
        assert pool.quality_multiplier == 1.5
        assert pool.adjusted_pool == 1_500_000

    def test_quality_multiplier_good(self) -> None:
        """Test good model gets 1x multiplier."""
        calculator = ContributionCalculator(base_reward_pool=1_000_000)

        evaluation = ModelEvaluation(
            benchmark_scores={"MMLU": 68.2, "GSM8K": 72.5},
            community_rating=0.75,
            sota_comparison=0.88,
        )

        pool = calculator.calculate_reward_pool(evaluation)
        assert pool.quality_multiplier == 1.0
        assert pool.adjusted_pool == 1_000_000

    def test_quality_multiplier_fair(self) -> None:
        """Test fair model gets 0.5x multiplier."""
        calculator = ContributionCalculator(base_reward_pool=1_000_000)

        evaluation = ModelEvaluation(
            benchmark_scores={"MMLU": 55.3, "GSM8K": 48.2},
            community_rating=0.55,
            sota_comparison=0.70,
        )

        pool = calculator.calculate_reward_pool(evaluation)
        assert pool.quality_multiplier == 0.5
        assert pool.adjusted_pool == 500_000

    def test_quality_multiplier_poor(self) -> None:
        """Test poor model gets 0x multiplier (no reward)."""
        calculator = ContributionCalculator(base_reward_pool=1_000_000)

        evaluation = ModelEvaluation(
            benchmark_scores={"MMLU": 35.2, "GSM8K": 28.5},
            community_rating=0.30,
            sota_comparison=0.45,
        )

        pool = calculator.calculate_reward_pool(evaluation)
        assert pool.quality_multiplier == 0.0
        assert pool.adjusted_pool == 0

    def test_reward_pool_split(self) -> None:
        """Test reward pool splitting after quality adjustment."""
        calculator = ContributionCalculator(base_reward_pool=1_000_000)

        evaluation = ModelEvaluation(
            benchmark_scores={"MMLU": 78.5, "GSM8K": 85.3},
            community_rating=0.90,
            sota_comparison=1.00,
        )

        pool = calculator.calculate_reward_pool(evaluation)

        # 1.5M total after quality multiplier
        assert pool.data_pool == 600_000  # 40%
        assert pool.compute_pool == 750_000  # 50%
        assert pool.paper_pool == 150_000  # 10%

    def test_compute_contribution_score(self) -> None:
        """Test contribution score calculation."""
        calculator = ContributionCalculator(base_reward_pool=1000)

        score = calculator.compute_contribution_score(
            compute_time=100.0,
            loss_reduction=0.5,
            vram_gb=24,
            total_vram=72,
        )

        expected = 100.0 * 0.5 * (24 / 72)
        assert score == expected

    def test_distribute_compute_rewards(self) -> None:
        """Test compute reward distribution."""
        calculator = ContributionCalculator(base_reward_pool=1_000_000)

        evaluation = ModelEvaluation(
            benchmark_scores={"MMLU": 78.5, "GSM8K": 85.3},
            community_rating=0.90,
            sota_comparison=1.00,
        )

        pool = calculator.calculate_reward_pool(evaluation)

        contributions = [
            Contribution(
                node_id="node1",
                compute_time=100.0,
                loss_reduction=0.5,
                vram_gb=24,
                verified=True,
            ),
            Contribution(
                node_id="node2",
                compute_time=100.0,
                loss_reduction=0.5,
                vram_gb=24,
                verified=True,
            ),
        ]

        rewards = calculator.distribute_compute_rewards(contributions, pool.compute_pool)

        assert "node1" in rewards
        assert "node2" in rewards
        # Equal contributions should get equal rewards
        assert rewards["node1"] == rewards["node2"]

    def test_unverified_contribution_zero_reward(self) -> None:
        """Test that unverified contributions get zero reward."""
        calculator = ContributionCalculator(base_reward_pool=1_000_000)

        evaluation = ModelEvaluation(
            benchmark_scores={"MMLU": 78.5, "GSM8K": 85.3},
            community_rating=0.90,
            sota_comparison=1.00,
        )

        pool = calculator.calculate_reward_pool(evaluation)

        contributions = [
            Contribution(
                node_id="node1",
                compute_time=100.0,
                loss_reduction=0.5,
                vram_gb=24,
                verified=False,
            ),
        ]

        rewards = calculator.distribute_compute_rewards(contributions, pool.compute_pool)

        assert rewards["node1"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
