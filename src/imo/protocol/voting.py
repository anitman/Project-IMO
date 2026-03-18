"""Vote-to-train mechanism for IMO governance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from imo.protocol.imo import IMO, IMOStatus


@dataclass
class VotingConfig:
    """Configuration for voting parameters."""

    quorum_threshold: float = 1000.0
    voting_period_hours: int = 168
    minimum_stake: float = 10.0
    vote_decay: float = 0.95


class VotingMechanism:
    """Manage voting for IMO approvals."""

    def __init__(self, config: VotingConfig | None = None):
        self.config = config or VotingConfig()

    def create_voting_period(
        self,
        imo: IMO,
        start_time: datetime | None = None,
    ) -> IMO:
        """Create voting period for an IMO."""
        start = start_time or datetime.now(timezone.utc)
        end = start + timedelta(hours=self.config.voting_period_hours)

        imo.voting_deadline = end.isoformat()
        imo.status = IMOStatus.VOTING
        imo.quorum_required = self.config.quorum_threshold

        return imo

    def cast_vote(
        self,
        imo: IMO,
        voter_id: str,
        vote: bool,
        stake: float,
    ) -> None:
        """Cast a vote on an IMO."""
        if imo.status != IMOStatus.VOTING:
            raise RuntimeError(f"IMO not in voting phase: {imo.status}")

        if datetime.now(timezone.utc).isoformat() > imo.voting_deadline:
            raise RuntimeError("Voting period has ended")

        if stake < self.config.minimum_stake:
            raise ValueError(f"Stake below minimum: {self.config.minimum_stake}")

        imo.add_vote(voter_id, vote, stake)

    def resolve_voting(self, imo: IMO) -> IMOStatus:
        """Resolve voting and determine outcome."""
        if imo.status != IMOStatus.VOTING:
            return imo.status

        if datetime.now(timezone.utc).isoformat() > imo.voting_deadline:
            if imo.quorum_reached():
                imo.status = IMOStatus.APPROVED
            else:
                imo.status = IMOStatus.FAILED

        return imo.status

    def get_voting_stats(self, imo: IMO) -> dict[str, Any]:
        """Get voting statistics for an IMO."""
        yes_votes = sum(v.stake for v in imo.votes if v.vote)
        no_votes = sum(v.stake for v in imo.votes if not v.vote)
        total_votes = len(imo.votes)

        return {
            "imo_id": imo.id,
            "status": imo.status.value,
            "yes_stake": yes_votes,
            "no_stake": no_votes,
            "total_stake": imo.total_stake,
            "quorum_required": imo.quorum_required,
            "quorum_reached": imo.quorum_reached(),
            "total_voters": total_votes,
            "voting_deadline": imo.voting_deadline,
            "time_remaining": self._calculate_time_remaining(imo),
        }

    def _calculate_time_remaining(self, imo: IMO) -> float:
        """Calculate time remaining in voting period (hours)."""
        if imo.status != IMOStatus.VOTING or not imo.voting_deadline:
            return 0.0

        deadline = datetime.fromisoformat(imo.voting_deadline)
        now = datetime.now(timezone.utc)
        delta = deadline - now

        return max(0, delta.total_seconds() / 3600)


class ReputationWeightedVoting(VotingMechanism):
    """Voting mechanism with reputation weighting."""

    def __init__(
        self,
        config: VotingConfig | None = None,
        reputation_scale: float = 1.0,
    ):
        super().__init__(config)
        self.reputation_scale = reputation_scale
        self.reputations: dict[str, float] = {}

    def set_reputation(self, voter_id: str, reputation: float) -> None:
        """Set reputation for a voter."""
        self.reputations[voter_id] = reputation

    def cast_vote(
        self,
        imo: IMO,
        voter_id: str,
        vote: bool,
        stake: float,
    ) -> None:
        """Cast a vote with reputation weighting."""
        reputation = self.reputations.get(voter_id, 1.0)
        weighted_stake = stake * (reputation**self.reputation_scale)

        super().cast_vote(imo, voter_id, vote, weighted_stake)
